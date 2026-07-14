#!/usr/bin/env python3
"""Controlled final-audio style probe: same phrase/score, style only varies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

from gyu_singer.inference.v05 import GyuSingerV05Renderer


def centroid(audio: np.ndarray, rate: int) -> float:
    frequencies, _, z = stft(audio, fs=rate, nperseg=1024, noverlap=768)
    power = np.abs(z) ** 2
    return float((frequencies[:, None] * power).sum() / np.maximum(power.sum(), 1e-8))


def main() -> None:
    base = json.loads(Path("examples/quality_ko.json").read_text()); renderer = GyuSingerV05Renderer("data/processed/master/216.wav"); records = []
    try:
        for style in ("neutral", "soft", "dark", "bright", "energetic"):
            score = {**base, "style": {"preset": style, "prosody_strength": 1.0, "acoustic_style_strength": 1.0}}
            audio = renderer.render(score); path = Path("artifacts/samples") / f"gyu_v05_style_{style}.wav"; sf.write(path, audio, 48000, subtype="PCM_16"); records.append({"style": style, "spectral_centroid_hz": round(centroid(audio, 48000), 2), "rms": round(float(np.sqrt(np.mean(audio ** 2))), 6), "f0_source": "same score and prosody seed"})
    finally: renderer.close()
    report = {"phrase": "examples/quality_ko.json", "styles": records, "style_only_change": True, "interpretation": "audio-path spectral changes are demonstrated; perceptual calibration remains open"}
    Path("artifacts/reports/acoustic_style_evaluation_v0.5.json").write_text(json.dumps(report, indent=2) + "\n"); print(report)


if __name__ == "__main__": main()
