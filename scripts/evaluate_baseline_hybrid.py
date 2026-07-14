#!/usr/bin/env python3
"""Identical-score baseline vs hybrid measurements using RMVPE, WavLM and Whisper."""
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

from gyu_singer.inference import HybridRenderer, load_hybrid_model
from gyu_singer.inference.codec import MossCodecDecoder
from gyu_singer.neural_renderer import NeuralRenderer

sys.path.insert(0, "data/cache/soulx-singer")
from preprocess.tools.f0_extraction import F0Extractor


SCORES = {
    "ko": ["하", "늘"], "en": ["la", "la"], "ja": ["あ", "い"],
}


def audio16(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return resample_poly(mono, 16000, rate).astype(np.float32) if rate != 16000 else mono


def embedding(model, extractor, audio: np.ndarray, device: str) -> np.ndarray:
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode(): value = model(**{key: value.to(device) for key, value in inputs.items()}).embeddings
    return torch.nn.functional.normalize(value, dim=-1).squeeze().cpu().numpy()


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    return round(float(np.dot(left, right)), 4)


def normalize(text: str) -> str:
    return re.sub(r"[^a-zA-Z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def levenshtein(expected: str, actual: str) -> float:
    expected, actual = normalize(expected), normalize(actual)
    if not expected or not actual: return 0.0
    previous = list(range(len(actual) + 1))
    for i, char in enumerate(expected, 1):
        current = [i]
        for j, other in enumerate(actual): current.append(min(current[-1] + 1, previous[j + 1] + 1, previous[j] + (char != other)))
        previous = current
    return round(1 - previous[-1] / max(len(expected), len(actual)), 4)


def metrics(path: str, score: dict, f0: F0Extractor, ref: np.ndarray, wavlm, extractor, asr, processor, device: str) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    observed = f0.process(path, verbose=False)
    times = np.arange(len(observed)) * .02
    target = np.zeros_like(observed)
    for note in score["notes"]:
        mask = (times >= note["start"]) & (times < note["start"] + note["duration"])
        target[mask] = 440 * 2 ** ((note["pitch"] - 69) / 12)
    voiced = observed > 0
    both = voiced & (target > 0)
    cents = 1200 * np.log2(observed[both] / target[both]) if both.any() else np.array([])
    correlation = float(np.corrcoef(np.log2(observed[both]), np.log2(target[both]))[0, 1]) if both.sum() > 3 and np.std(target[both]) > 1e-6 else None
    boundary = int(score["notes"][1]["start"] * rate)
    win = int(.04 * rate)
    before, after = mono[max(0, boundary - win):boundary], mono[boundary:min(len(mono), boundary + win)]
    jump = abs(np.log(np.sqrt(np.mean(before**2)) + 1e-7) - np.log(np.sqrt(np.mean(after**2)) + 1e-7))
    inputs = processor(audio16(path), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode(): tokens = asr.generate(inputs.input_features.to(device, torch.float16 if device.startswith("cuda") else torch.float32), language=score["language"], task="transcribe", max_new_tokens=64)
    transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    held = observed[(times >= .65) & (times < 1.15) & voiced]
    return {"rmvpe_f0_correlation": None if correlation is None or not np.isfinite(correlation) else round(correlation, 4), "note_pitch_mae_cents": round(float(np.mean(abs(cents))), 2) if len(cents) else None,
            "voiced_accuracy": round(float(np.mean(voiced == (target > 0))), 4), "boundary_energy_jump": round(float(jump), 4), "speaker_similarity_wavlm": similarity(ref, embedding(wavlm, extractor, audio16(path), device)),
            "asr_transcript": transcript, "asr_lyric_similarity": levenshtein("".join(note["lyric"] for note in score["notes"]), transcript), "held_note_f0_cv": round(float(np.std(held) / np.mean(held)), 4) if len(held) > 2 else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline-render", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("artifacts/samples").mkdir(exist_ok=True)
    baseline = None if args.skip_baseline_render else NeuralRenderer("checkpoints/gyu_moss_nano_sft/checkpoint-last", "data/cache/moss-audio-tokenizer-nano", "data/source/Korea Digital Media High School 215.m4a")
    hybrid = HybridRenderer(load_hybrid_model("checkpoints/gyu_hybrid_v0.2.pt", device), MossCodecDecoder("data/cache/moss-audio-tokenizer-nano", device), "data/processed/master/216.wav")
    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    ref = embedding(wavlm, extractor, audio16("data/processed/master/216.wav"), device)
    processor = AutoProcessor.from_pretrained("data/cache/whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained("data/cache/whisper-large-v3-turbo", dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    f0 = F0Extractor("data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt", device=device, target_sr=24000, hop_size=480, verbose=False)
    result = {"f0_extractor": "SoulX RMVPE", "scores": {}}
    for language, lyrics in SCORES.items():
        score = {"language": language, "tempo": 120, "sample_rate": 48000, "notes": [{"pitch": 60, "start": 0.0, "duration": .6, "lyric": lyrics[0]}, {"pitch": 64, "start": .6, "duration": .6, "lyric": lyrics[1]}]}
        paths = {"vocalizer_baseline": f"artifacts/samples/baseline_{language}.wav", "hybrid_svs": f"artifacts/samples/hybrid_{language}.wav"}
        if baseline: sf.write(paths["vocalizer_baseline"], baseline.render(score), 48000)
        torch.manual_seed(21)
        sf.write(paths["hybrid_svs"], hybrid.render(score), 48000)
        result["scores"][language] = {name: metrics(path, score, f0, ref, wavlm, extractor, asr, processor, device) | {"path": path} for name, path in paths.items()}
    Path("artifacts/reports/baseline_hybrid_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
