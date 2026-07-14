#!/usr/bin/env python3
"""Objective quality gate for the score-conditioned SoulX phrase probe."""
from __future__ import annotations

import json
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, "data/cache/soulx-singer")
from preprocess.tools.f0_extraction import F0Extractor


PROBE_CASES = {
    "ko": ("artifacts/samples/soulx_score_ko.wav", "하늘을 향해 노래해\n작은 빛을 따라가", "artifacts/samples/soulx_score_ko.score.json"),
    "en": ("artifacts/samples/soulx_score_en.wav", "Sing into the open sky\nFollow the golden light", "artifacts/samples/soulx_score_en.score.json"),
    "ja": ("artifacts/samples/soulx_score_ja.wav", "空へ向かい歌おう\n小さな光を追う", "artifacts/samples/soulx_score_ja.score.json"),
}
RUNTIME_CASES = {"ko": ("/tmp/gyu-soulx-quality-ko.wav", "하늘을 향해 노래해 작은 빛을 따라가", "examples/quality_ko.json"), "en": ("/tmp/gyu-soulx-quality-en.wav", "Sing into the open sky Follow the golden light", "examples/quality_en.json"), "ja": ("/tmp/gyu-soulx-quality-ja.wav", "空へ向かい歌おう 小さな光を追う", "examples/quality_ja.json")}
HELDOUT_CASES = {"ko": ("/tmp/gyu-heldout-ko.wav", "바람이 불어와 마음을 감싸줘", "examples/heldout_ko.json"), "en": ("/tmp/gyu-heldout-en.wav", "Carry my voice across the quiet river", "examples/heldout_en.json"), "ja": ("/tmp/gyu-heldout-ja.wav", "新しい歌を風に乗せて届ける", "examples/heldout_ja.json")}


def audio16(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return resample_poly(mono, 16000, rate).astype(np.float32) if rate != 16000 else mono


def normalized(text: str) -> str:
    return re.sub(r"[^a-zA-Z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def lyric_similarity(expected: str, actual: str) -> float:
    expected, actual = normalized(expected), normalized(actual)
    if not expected or not actual:
        return 0.0
    row = list(range(len(actual) + 1))
    for i, char in enumerate(expected, 1):
        next_row = [i]
        for j, other in enumerate(actual):
            next_row.append(min(next_row[-1] + 1, row[j + 1] + 1, row[j] + (char != other)))
        row = next_row
    return round(1 - row[-1] / max(len(expected), len(actual)), 4)


def speaker_embedding(model, extractor, path: str, device: str) -> np.ndarray:
    inputs = extractor(audio16(path), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        value = model(**{key: val.to(device) for key, val in inputs.items()}).embeddings
    return torch.nn.functional.normalize(value, dim=-1).squeeze().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--runtime-smoke", action="store_true"); parser.add_argument("--heldout", action="store_true"); args = parser.parse_args()
    if args.runtime_smoke and args.heldout: parser.error("choose one case set")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f0 = F0Extractor("data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt", device=device, target_sr=24000, hop_size=480, verbose=False)
    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    reference = speaker_embedding(wavlm, extractor, "data/processed/master/216.wav", device)
    processor = AutoProcessor.from_pretrained("data/cache/whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained("data/cache/whisper-large-v3-turbo", dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    results = {"gate": {"f0_correlation_min": 0.9, "pitch_mae_cents_max": 100, "held_note_f0_cv_max": 0.1, "asr_lyric_similarity_min": 0.5}, "cases": {}}
    cases = HELDOUT_CASES if args.heldout else RUNTIME_CASES if args.runtime_smoke else PROBE_CASES
    for language, (path, text, score_path) in cases.items():
        observed = f0.process(path, verbose=False)
        score = json.loads(Path(score_path).read_text())
        target = np.zeros_like(observed)
        for note in score["notes"]:
            start = round(float(note["start"]) * 50); end = round((float(note["start"]) + float(note["duration"])) * 50)
            target[start:end] = 440 * 2 ** ((int(note["pitch"]) - 69) / 12)
        voiced = observed > 0
        both = voiced & (target > 0)
        cents = 1200 * np.log2(observed[both] / target[both])
        corr = float(np.corrcoef(np.log2(observed[both]), np.log2(target[both]))[0, 1]) if both.sum() > 3 else float("nan")
        cvs = []
        for note in score["notes"]:
            start = int((float(note["start"]) + float(note["duration"]) * .2) * 50)
            end = int((float(note["start"]) + float(note["duration"]) * .8) * 50)
            held = observed[start:end]; held = held[held > 0]
            if len(held) > 2: cvs.append(float(np.std(held) / np.mean(held)))
        asr_in = processor(audio16(path), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(asr_in.input_features.to(device, torch.float16 if device == "cuda" else torch.float32), language=language, task="transcribe", max_new_tokens=128)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        metrics = {
            "source_phrase": text, "asr_transcript": transcript,
            "rmvpe_f0_correlation": round(corr, 4), "note_pitch_mae_cents": round(float(np.mean(abs(cents))), 2),
            "voiced_fraction": round(float(voiced.mean()), 4), "held_note_f0_cv": round(float(np.median(cvs)), 4),
            "speaker_similarity_wavlm": round(float(np.dot(reference, speaker_embedding(wavlm, extractor, path, device))), 4),
            "asr_lyric_similarity": lyric_similarity(text, transcript),
        }
        metrics["pass"] = all((metrics["rmvpe_f0_correlation"] >= .9, metrics["note_pitch_mae_cents"] <= 100, metrics["held_note_f0_cv"] <= .1, metrics["asr_lyric_similarity"] >= .5))
        results["cases"][language] = metrics
    results["pass"] = all(case["pass"] for case in results["cases"].values())
    output = Path("artifacts/reports/soulx_heldout_smoke.json" if args.heldout else "artifacts/reports/soulx_runtime_smoke.json" if args.runtime_smoke else "artifacts/reports/soulx_score_probe.json"); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
