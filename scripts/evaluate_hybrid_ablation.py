#!/usr/bin/env python3
"""Short A/B teacher-distillation check using identical phrase scores and WavLM similarity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

from gyu_singer.inference import HybridRenderer, load_hybrid_model
from gyu_singer.inference.codec import MossCodecDecoder


SCORES = {
    "ko": ["하", "늘"], "en": ["la", "la"], "ja": ["あ", "い"],
}


def audio16(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return resample_poly(mono, 16000, rate).astype(np.float32) if rate != 16000 else mono


def embed(model, extractor, audio: np.ndarray, device: str) -> np.ndarray:
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        value = model(**{key: value.to(device) for key, value in inputs.items()}).embeddings
    return torch.nn.functional.normalize(value, dim=-1).squeeze().cpu().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = MossCodecDecoder("data/cache/moss-audio-tokenizer-nano", device)
    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    reference = embed(wavlm, extractor, audio16("data/processed/master/216.wav"), device)
    output = {"metric": "WavLM cosine to GYU reference; short ablation, not intelligibility evidence", "runs": {}}
    for label, checkpoint in {"no_teacher": "checkpoints/gyu_hybrid_ablation_no_teacher.pt", "with_teacher": "checkpoints/gyu_hybrid_ablation_teacher.pt"}.items():
        renderer = HybridRenderer(load_hybrid_model(checkpoint, device), codec, "data/processed/master/216.wav")
        scores = {}
        for language, lyrics in SCORES.items():
            score = {"language": language, "tempo": 120, "style": {"preset": "neutral"}, "notes": [{"id": "n1", "pitch": 60, "start_beat": 0, "duration_beats": 1, "lyric": lyrics[0]}, {"id": "n2", "pitch": 64, "start_beat": 1, "duration_beats": 1, "lyric": lyrics[1]}]}
            torch.manual_seed(11)
            path = f"artifacts/samples/ablation_{label}_{language}.wav"
            sf.write(path, renderer.render(score), 48000)
            vector = embed(wavlm, extractor, audio16(path), device)
            scores[language] = round(float(np.dot(reference, vector)), 4)
        output["runs"][label] = scores
    Path("artifacts/reports/hybrid_teacher_ablation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
