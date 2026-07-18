#!/usr/bin/env python3
"""Gate local PJS DiffSinger rapid-song probes without publishing lyrics/audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CACHE / "soulx-singer"))

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lyric_similarity(expected: list[str], transcript: str) -> float:
    return max(
        SequenceMatcher(None, normalized(value), normalized(transcript)).ratio()
        for value in expected
    )


def f0_summary(values: np.ndarray) -> dict:
    voiced = np.asarray(values)[np.asarray(values) > 1]
    if not len(voiced):
        return {"hz_p05": None, "hz_median": None, "hz_p95": None,
                "midi_p05": None, "midi_median": None, "midi_p95": None}
    midi = 69 + 12 * np.log2(voiced / 440)
    return {
        "hz_p05": round(float(np.percentile(voiced, 5)), 2),
        "hz_median": round(float(np.median(voiced)), 2),
        "hz_p95": round(float(np.percentile(voiced, 95)), 2),
        "midi_p05": round(float(np.percentile(midi, 5)), 2),
        "midi_median": round(float(np.median(midi)), 2),
        "midi_p95": round(float(np.percentile(midi, 95)), 2),
    }


def teacher_forced_nll(processor, model, audio: np.ndarray, expected: list[str]) -> float:
    inputs = processor(audio=audio, sampling_rate=16_000, return_tensors="pt",
                       return_attention_mask=True)
    losses = []
    for text in expected:
        labels = processor.tokenizer(text, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            loss = model(input_features=inputs.input_features.cuda().half(),
                         attention_mask=inputs.attention_mask.cuda(), labels=labels).loss
        losses.append(float(loss))
    return min(losses)


def passes_gate(row: dict, reference: dict | None = None) -> bool:
    """Require a free transcript; teacher-forced likelihood is diagnostic only."""
    spike_ok = reference is None or (
        row["hf_spike_p99_over_median"] <= 2 * reference["hf_spike_p99_over_median"]
    )
    voicing_ok = (
        row["voicing_accuracy"] >= .8
        if "voicing_accuracy" in row
        else .8 <= row["observed_voiced_ratio"] <= .97
    )
    return bool(
        row["asr_lyric_similarity"] >= .8
        and row["pitch_p90_abs_cents"] <= 100
        and row["gross_error_over_600_cents"] <= .05
        and voicing_ok
        and row["clip_fraction"] == 0
        and spike_ok
    )


def _equal_hop_f0(target: np.ndarray, observed: np.ndarray, max_frame_delta: int = 1):
    if abs(len(target) - len(observed)) > max_frame_delta:
        raise ValueError(
            f"F0 grid length mismatch exceeds {max_frame_delta} frame: "
            f"target={len(target)}, observed={len(observed)}"
        )
    frames = min(len(target), len(observed))
    return target[:frames], observed[:frames]


def pitch_errors(target: np.ndarray, observed: np.ndarray, *, max_frame_delta: int = 1) -> np.ndarray:
    """Compare equal-hop F0 without inventing pitch across unvoiced boundaries."""
    target, observed = _equal_hop_f0(target, observed, max_frame_delta)
    both = (observed > 1) & (target > 1)
    if not np.any(both):
        raise ValueError("candidate and target have no jointly voiced F0 frames")
    return np.abs(1200 * np.log2(observed[both] / target[both]))


def voicing_accuracy(target: np.ndarray, observed: np.ndarray, *, max_frame_delta: int = 1) -> float:
    target, observed = _equal_hop_f0(target, observed, max_frame_delta)
    return float(np.mean((target > 1) == (observed > 1)))


def _stats(values: dict[str, float]) -> dict:
    data = list(values.values())
    return {
        "values": values,
        "mean": round(float(np.mean(data)), 5),
        "min": min(data),
        "max": max(data),
    }


def similarity_summary(
    references: dict[str, tuple[np.ndarray, np.ndarray]],
    current: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Keep per-reference speaker evidence instead of hiding its distribution."""
    wavlm = {
        name: round(float(np.dot(value[0], current[0])), 5)
        for name, value in references.items()
    }
    ecapa = {
        name: round(float(np.dot(value[1], current[1])), 5)
        for name, value in references.items()
    }
    return {
        "reference_count": len(references),
        "wavlm": _stats(wavlm),
        "ecapa": _stats(ecapa),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=Path, required=True)
    parser.add_argument("--expected-text", action="append", required=True,
                        help="Expected lyric spelling; repeat for kanji/kana aliases")
    parser.add_argument("--candidate", action="append", required=True,
                        help="LABEL=/absolute/or/repository/path.wav")
    parser.add_argument("--reference-audio", type=Path,
                        help="Known-correct lyric audio used to calibrate the ASR/NLL gate")
    parser.add_argument("--identity-reference", action="append", type=Path, default=[],
                        help="Repeat for each real target-speaker reference")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/reports/diffsinger_pjs_rapid_evaluation.json")
    args = parser.parse_args()

    target = np.asarray(json.loads(args.ds.read_text())[0]["f0_seq"].split(), dtype=np.float32)
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
    )

    reference_nll = None
    reference_asr_similarity = None
    reference_metrics = None
    if args.reference_audio is not None:
        reference_path = args.reference_audio if args.reference_audio.is_absolute() else ROOT / args.reference_audio
        reference_audio = audio16(reference_path)
        reference_inputs = processor(audio=reference_audio, sampling_rate=16_000,
                                     return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            reference_ids = whisper.generate(
                reference_inputs.input_features.cuda().half(),
                attention_mask=reference_inputs.attention_mask.cuda() if "attention_mask" in reference_inputs else None,
                language="ja", task="transcribe", max_new_tokens=64,
            )
        reference_transcript = processor.batch_decode(reference_ids, skip_special_tokens=True)[0]
        reference_asr_similarity = lyric_similarity(args.expected_text, reference_transcript)
        reference_nll = teacher_forced_nll(processor, whisper, reference_audio, args.expected_text)
        reference_f0 = np.asarray(extractor.process(str(reference_path), verbose=False), dtype=np.float32)
        reference_metrics = {
            "audio_path": str(reference_path.relative_to(ROOT)),
            "duration_seconds": round(len(reference_audio) / 16_000, 4),
            "free_asr_transcript": reference_transcript,
            "f0": f0_summary(reference_f0),
            "observed_voiced_ratio": round(float(np.mean(reference_f0 > 1)), 4),
        } | acoustics(reference_path)

    rows = []
    for value in args.candidate:
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        inputs = processor(audio=audio16(path), sampling_rate=16_000, return_tensors="pt",
                           return_attention_mask=True)
        with torch.inference_mode():
            ids = whisper.generate(
                inputs.input_features.cuda().half(),
                attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
                language="ja", task="transcribe",
                max_new_tokens=64,
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
        cents = pitch_errors(target, observed)
        metrics = {
            "label": label,
            "audio_path": str(path.relative_to(ROOT)),
            "audio_sha256": sha256(path),
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(lyric_similarity(args.expected_text, transcript), 4),
            "teacher_forced_lyric_nll": round(
                teacher_forced_nll(processor, whisper, audio16(path), args.expected_text), 4
            ),
            "pitch_median_abs_cents": round(float(np.median(cents)), 2),
            "pitch_p90_abs_cents": round(float(np.percentile(cents, 90)), 2),
            "gross_error_over_600_cents": round(float(np.mean(cents > 600)), 4),
            "observed_voiced_ratio": round(float(np.mean(observed > 1)), 4),
            "voicing_accuracy": round(voicing_accuracy(target, observed), 4),
            "f0": f0_summary(observed),
        } | acoustics(path)
        metrics["pass"] = passes_gate(metrics, reference_metrics)
        rows.append(metrics)

    identity_reference_paths = []
    if args.identity_reference:
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        del whisper
        torch.cuda.empty_cache()
        feature_extractor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
        wavlm = AutoModelForAudioXVector.from_pretrained(
            CACHE / "wavlm-base-plus-sv"
        ).cuda().eval()
        ecapa = EncoderClassifier.from_hparams(
            source=str(CACHE / "spkrec-ecapa-voxceleb"),
            savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cuda:0"},
        )

        def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
            audio = audio16(path)
            values = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
            with torch.inference_mode():
                first = wavlm(
                    **{key: value.cuda() for key, value in values.items()}
                ).embeddings
                second = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
            first = torch.nn.functional.normalize(first, dim=-1).squeeze().cpu().numpy()
            second = second.squeeze().cpu().numpy()
            second /= max(np.linalg.norm(second), 1e-8)
            return first, second

        references = {}
        for raw_path in args.identity_reference:
            path = raw_path if raw_path.is_absolute() else ROOT / raw_path
            name = str(path.relative_to(ROOT))
            identity_reference_paths.append(name)
            references[name] = speaker(path)
        for row in rows:
            row["identity_similarity"] = similarity_summary(
                references, speaker(ROOT / row["audio_path"])
            )

    def candidate_key(row: dict) -> tuple:
        identity = row.get("identity_similarity")
        wavlm_mean = identity["wavlm"]["mean"] if identity and row["pass"] else -1.0
        ecapa_mean = identity["ecapa"]["mean"] if identity and row["pass"] else -1.0
        return (
            row["pass"], wavlm_mean, ecapa_mean,
            row["asr_lyric_similarity"], -row["pitch_p90_abs_cents"],
        )

    candidate = max(rows, key=candidate_key)
    source_pass = reference_asr_similarity is None or reference_asr_similarity >= .8
    report = {
        "status": "source_probe_pass_human_pending" if candidate["pass"] and source_pass else "source_probe_reject",
        "selected": candidate["label"] if candidate["pass"] and source_pass else None,
        "best_diagnostic_candidate": candidate["label"],
        "gate": {
            "asr_lyric_similarity_min": .8,
            "teacher_forced_lyric_nll": "diagnostic_only_never_a_pass_substitute",
            "pitch_p90_abs_cents_max": 100,
            "gross_error_over_600_cents_max": .05,
            "observed_voiced_ratio_min": .8,
            "observed_voiced_ratio_max": .97,
            "voicing_accuracy_min_when_target_grid_available": .8,
            "clip_fraction": 0,
            "hf_spike_p99_over_reference_max": 2.0,
        },
        "reference_calibration": {
            "used": args.reference_audio is not None,
            "free_asr_similarity": (round(reference_asr_similarity, 4)
                                    if reference_asr_similarity is not None else None),
            "teacher_forced_lyric_nll": (round(reference_nll, 4)
                                         if reference_nll is not None else None),
            "free_asr_reference_pass": (
                reference_asr_similarity is not None and reference_asr_similarity >= .8
            ),
            "waveform_analysis": reference_metrics,
        },
        "target_f0": f0_summary(target),
        "expected_text_sha256": [
            hashlib.sha256(normalized(value).encode("utf-8")).hexdigest()
            for value in args.expected_text
        ],
        "expected_text_characters": [len(normalized(value)) for value in args.expected_text],
        "rows": rows,
        "identity_reference_paths": identity_reference_paths,
        "identity_adaptation_allowed": bool(candidate["pass"] and source_pass),
        "release_allowed": False,
        "interpretation": (
            "Every candidate must pass waveform/F0 checks and a free Whisper transcript. "
            "Teacher-forced NLL cannot rescue an unintelligible candidate; human listening remains mandatory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
