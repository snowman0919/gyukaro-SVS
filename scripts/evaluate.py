#!/usr/bin/env python3
"""Score generated teacher audio without admitting it into the real corpus."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def resample(audio: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = np.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def normalized(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def levenshtein_similarity(expected: str, actual: str) -> float:
    expected, actual = normalized(expected), normalized(actual)
    if not expected or not actual:
        return 0.0
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return round(1 - previous[-1] / max(len(expected), len(actual)), 4)


def script_language_score(text: str, language: str) -> float:
    language = {"Korean": "ko", "English": "en", "Japanese": "ja"}.get(language, language)
    compact = normalized(text)
    if not compact:
        return 0.0
    if language == "ko":
        matched = sum("가" <= char <= "힣" for char in compact)
    elif language == "ja":
        matched = sum(("ぁ" <= char <= "ん") or ("ァ" <= char <= "ン") or ("一" <= char <= "龯") for char in compact)
    else:
        matched = sum(char.isascii() and char.isalpha() for char in compact)
    return round(matched / len(compact), 4)


def audio_metrics(path: str) -> tuple[np.ndarray, int, dict]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    silence = float(np.mean(np.abs(mono) < 0.01))
    f0 = librosa.yin(mono, fmin=65, fmax=1_047, sr=rate, frame_length=2048, hop_length=512)
    f0 = f0[np.isfinite(f0)]
    metrics = {
        "sample_rate": rate, "channels": audio.shape[1], "duration_sec": round(len(mono) / rate, 4),
        "peak": round(peak, 6), "rms": round(rms, 6), "silence_ratio": round(silence, 6),
        "f0_median_hz": round(float(np.median(f0)), 3) if len(f0) else None,
        "acoustic_pass": bool(0.2 <= len(mono) / rate <= 30 and peak < 0.995 and rms > 0.002 and silence < 0.75),
    }
    return mono, rate, metrics


class Evaluators:
    def __init__(self, device: str, asr_path: str, wavlm_path: str, ecapa_path: str):
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device
        self.asr_processor = AutoProcessor.from_pretrained(asr_path)
        self.asr = AutoModelForSpeechSeq2Seq.from_pretrained(asr_path, dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device).eval()
        self.wavlm_processor = AutoFeatureExtractor.from_pretrained(wavlm_path)
        self.wavlm = AutoModelForAudioXVector.from_pretrained(wavlm_path).to(device).eval()
        self.ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=ecapa_path, run_opts={"device": device})

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        inputs = self.asr_processor(audio, sampling_rate=16_000, return_tensors="pt")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        with torch.inference_mode():
            tokens = self.asr.generate(inputs.input_features.to(self.device, dtype), language=language, task="transcribe", max_new_tokens=128)
        return self.asr_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def wavlm_embedding(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.wavlm_processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            embedding = self.wavlm(**{key: value.to(self.device) for key, value in inputs.items()}).embeddings
        return torch.nn.functional.normalize(embedding, dim=-1).squeeze().cpu().numpy()

    def ecapa_embedding(self, audio: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.ecapa.encode_batch(waveform).squeeze()
        return torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return round(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))), 4)


def language_code(language: str) -> str:
    return {"Korean": "ko", "English": "en", "Japanese": "ja"}.get(language, language)


def agreement_peers(rows: list[dict], row: dict) -> list[dict]:
    return [peer for peer in rows if peer is not row and peer["teacher"] != row["teacher"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/teacher_executed.jsonl")
    parser.add_argument("--output", default="artifacts/eval/teacher_scored.jsonl")
    parser.add_argument("--asr-model", default="data/cache/whisper-large-v3-turbo")
    parser.add_argument("--wavlm-model", default="data/cache/wavlm-base-plus-sv")
    parser.add_argument("--ecapa-cache", default="data/cache/spkrec-ecapa-voxceleb")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluators(device, args.asr_model, args.wavlm_model, args.ecapa_cache)
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    generated_wavlm: dict[str, np.ndarray] = {}
    for row in rows:
        source_index = int(row["reference_ids"][0].rsplit("_", 1)[1])
        reference_path = f"data/processed/master/{source_index}.wav"
        if reference_path not in references:
            reference_audio, reference_rate, _ = audio_metrics(reference_path)
            reference_audio = resample(reference_audio, reference_rate)
            references[reference_path] = (evaluator.wavlm_embedding(reference_audio), evaluator.ecapa_embedding(reference_audio))
        audio, rate, metrics = audio_metrics(row["output_path"])
        audio16 = resample(audio, rate)
        transcript = evaluator.transcribe(audio16, language_code(row["language"]))
        wavlm, ecapa = references[reference_path]
        generated_wavlm[row["output_path"]] = evaluator.wavlm_embedding(audio16)
        row.update(metrics | {
            "asr_model": "openai/whisper-large-v3-turbo",
            "asr_transcript": transcript,
            "content_score": levenshtein_similarity(row["text"], transcript),
            "language_score": script_language_score(transcript, row["language"]),
            "speaker_score": cosine(generated_wavlm[row["output_path"]], wavlm),
            "speaker_score_2": cosine(evaluator.ecapa_embedding(audio16), ecapa),
        })
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)
    for row in rows:
        peers = agreement_peers(grouped[row["id"]], row)
        row["teacher_agreement_score"] = round(float(np.mean([cosine(generated_wavlm[row["output_path"]], generated_wavlm[peer["output_path"]]) for peer in peers])), 4) if peers else None
        minimums = [row["acoustic_pass"], row["content_score"] >= 0.55, row["language_score"] >= 0.8, row["speaker_score"] >= 0.1, row["speaker_score_2"] >= 0.1]
        row["overall_confidence"] = round(float(np.mean([row["content_score"], row["language_score"], max(row["speaker_score"], 0), max(row["speaker_score_2"], 0)])), 4)
        row["quality_status"] = "teacher_gate_pass_unadmitted" if all(minimums) else "review_required"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"scored={len(rows)} accepted={sum(row['quality_status'] == 'teacher_gate_pass_unadmitted' for row in rows)} output={output}")


if __name__ == "__main__":
    main()
