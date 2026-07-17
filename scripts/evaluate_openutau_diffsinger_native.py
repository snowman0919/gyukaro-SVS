#!/usr/bin/env python3
"""Evaluate an official OpenUtau DiffSinger render with waveform, F0 and free ASR."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CACHE / "soulx-singer"))

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


UNVOICED = {
    "AP", "SP", "c_ja", "h_ja", "k_ja", "p_ja", "s_ja", "t_ja", "ts_ja",
    "tɕ_ja", "ç_ja", "ɕ_ja", "ɸ_ja", "ʔ_ja", "i̥_ja", "ɨ̥_ja", "ɯ̥_ja",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sung_region(metrics: dict) -> tuple[float, float]:
    phones = [phone for phone in metrics["phoneme_timeline"]
              if phone["phoneme"] not in {"AP", "SP"}]
    if not phones:
        raise ValueError("no sung phoneme region")
    return min(phone["start_ms"] for phone in phones), max(phone["end_ms"] for phone in phones)


def region_acoustics(path: Path, start_ms: float, end_ms: float) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    start = max(0, round(start_ms * rate / 1000))
    end = min(len(audio), round(end_ms * rate / 1000))
    if end <= start:
        raise ValueError(f"invalid sung region: {start_ms}..{end_ms} ms")
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    temporary_path = Path(temporary.name)
    try:
        sf.write(temporary_path, audio[start:end], rate, subtype="FLOAT")
        return acoustics(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def identity_similarity(candidate: Path, reference: Path) -> dict:
    feature_extractor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"),
        savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )

    def embed(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio16(path)
        values = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            first = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            second = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        first = torch.nn.functional.normalize(first, dim=-1).squeeze().cpu().numpy()
        second = second.squeeze().cpu().numpy()
        second /= max(np.linalg.norm(second), 1e-8)
        return first, second

    reference_embedding = embed(reference)
    candidate_embedding = embed(candidate)
    return {
        "identity_reference": str(reference.relative_to(ROOT)),
        "identity_reference_sha256": sha256(reference),
        "wavlm_to_identity_reference": round(
            float(np.dot(reference_embedding[0], candidate_embedding[0])), 5),
        "ecapa_to_identity_reference": round(
            float(np.dot(reference_embedding[1], candidate_embedding[1])), 5),
    }


def score_f0(metrics: dict, frames: int, hop_ms: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(frames, dtype=np.float32)
    phoneme_present = np.zeros(frames, dtype=bool)
    centers = (np.arange(frames) + .5) * hop_ms
    for phone in metrics["phoneme_timeline"]:
        selected = (centers >= phone["start_ms"]) & (centers < phone["end_ms"])
        phoneme_present[selected] = True
        if phone["phoneme"] not in UNVOICED:
            target[selected] = 440 * 2 ** ((phone["tone"] - 69) / 12)
    return target, phoneme_present


def f0_metrics(target: np.ndarray, observed: np.ndarray, phoneme_present: np.ndarray) -> dict:
    target_voiced = target > 1
    observed_voiced = observed > 1
    both = target_voiced & observed_voiced
    cents = np.abs(1200 * np.log2(observed[both] / target[both])) if both.any() else np.array([])
    true_positive = int(np.sum(both))
    precision = true_positive / max(int(np.sum(observed_voiced & phoneme_present)), 1)
    recall = true_positive / max(int(np.sum(target_voiced)), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    unvoiced = (~target_voiced) & phoneme_present
    return {
        "target_voiced_ratio": round(float(np.mean(target_voiced)), 4),
        "observed_voiced_ratio": round(float(np.mean(observed_voiced)), 4),
        "voicing_accuracy": round(float(np.mean(target_voiced == observed_voiced)), 4),
        "voicing_precision_in_phonemes": round(precision, 4),
        "voicing_recall": round(recall, 4),
        "voicing_f1": round(f1, 4),
        "unvoiced_phoneme_false_voiced_ratio": round(
            float(np.mean(observed_voiced[unvoiced])) if unvoiced.any() else 0.0, 4),
        "pitch_median_abs_cents": round(float(np.median(cents)), 2) if cents.size else None,
        "pitch_p90_abs_cents": round(float(np.percentile(cents, 90)), 2) if cents.size else None,
        "gross_error_over_600_cents": round(float(np.mean(cents > 600)), 4) if cents.size else None,
    }


def review_plot(audio_path: Path, target: np.ndarray, observed: np.ndarray, output: Path) -> None:
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    time = np.arange(len(mono)) / rate
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time, mono, linewidth=.5)
    axes[0].set_ylabel("waveform")
    spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=2048, hop_length=256)), ref=np.max)
    librosa.display.specshow(spectrum, sr=rate, hop_length=256, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_ylim(0, 16_000)
    axes[1].set_ylabel("spectrogram")
    f0_time = np.arange(len(observed)) * .02
    axes[2].plot(f0_time, np.where(target > 1, target, np.nan), label="score target", linewidth=1.5)
    axes[2].plot(f0_time, np.where(observed > 1, observed, np.nan), label="RMVPE observed", linewidth=1)
    axes[2].set_ylabel("F0 Hz")
    axes[2].set_xlabel("seconds")
    axes[2].legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--openutau-metrics", type=Path, required=True)
    parser.add_argument("--expected-text", action="append", required=True)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--identity-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio_path = args.audio.resolve()
    output_path = args.output.resolve()
    openutau = json.loads(args.openutau_metrics.read_text())
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
    )
    observed = np.asarray(extractor.process(str(audio_path), verbose=False), dtype=np.float32)
    target, phoneme_present = score_f0(openutau, len(observed))
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16).cuda().eval()
    inputs = processor(audio=audio16(audio_path), sampling_rate=16_000, return_tensors="pt",
                       return_attention_mask=True)
    with torch.inference_mode():
        ids = whisper.generate(
            inputs.input_features.cuda().half(),
            attention_mask=inputs.attention_mask.cuda(),
            language=args.language, task="transcribe", max_new_tokens=96,
        )
    transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
    similarity = max(SequenceMatcher(None, normalized(text), normalized(transcript)).ratio()
                     for text in args.expected_text)
    plot = output_path.with_suffix(".png")
    review_plot(audio_path, target, observed, plot)
    start_ms, end_ms = sung_region(openutau)
    whole_file = acoustics(audio_path)
    active_region = region_acoustics(audio_path, start_ms, end_ms)
    row = {
        "audio_path": str(audio_path.relative_to(ROOT)),
        "audio_sha256": sha256(audio_path),
        "free_whisper_transcript": transcript,
        "asr_lyric_similarity": round(similarity, 4),
        "waveform_whole_file": whole_file,
        "sung_region_ms": [round(start_ms, 3), round(end_ms, 3)],
        "waveform_sung_region": active_region,
    } | whole_file | f0_metrics(target, observed, phoneme_present)
    if args.identity_reference is not None:
        del whisper
        torch.cuda.empty_cache()
        reference = args.identity_reference.resolve()
        row |= identity_similarity(audio_path, reference)
    automated = bool(
        similarity >= .8
        and row["pitch_p90_abs_cents"] is not None and row["pitch_p90_abs_cents"] <= 100
        and row["gross_error_over_600_cents"] <= .05
        and row["voicing_f1"] >= .8
        and row["unvoiced_phoneme_false_voiced_ratio"] <= .2
        and row["clip_fraction"] == 0
    )
    report = {
        "status": "automated_gate_pass_human_pending" if automated else "automated_gate_fail",
        "official_openutau_diffsinger_renderer": openutau["official_openutau_diffsinger_renderer"],
        "expected_text_sha256": [hashlib.sha256(normalized(text).encode()).hexdigest()
                                 for text in args.expected_text],
        "openutau": openutau,
        "quality": row,
        "review_plot": str(plot.relative_to(ROOT)),
        "release_allowed": False,
        "human_listening_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
