#!/usr/bin/env python3
"""Measure whether an SVC pilot preserved its target duration and F0 contour."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def f0_contour(audio: np.ndarray, rate: int) -> np.ndarray:
    return librosa.yin(audio, fmin=65, fmax=1047, sr=rate, frame_length=2048, hop_length=512)


def contour_correlation(target: np.ndarray, output: np.ndarray) -> float:
    target = np.log2(target[np.isfinite(target) & (target > 0)])
    output = np.log2(output[np.isfinite(output) & (output > 0)])
    if len(target) < 3 or len(output) < 3:
        return 0.0
    target = np.interp(np.linspace(0, len(target) - 1, len(output)), np.arange(len(target)), target)
    return round(float(np.corrcoef(target, output)[0, 1]), 4)


def metrics(path: str) -> tuple[np.ndarray, int, float]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return f0_contour(mono, rate), rate, len(mono) / rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    target_f0, target_rate, target_duration = metrics(args.target)
    output_f0, output_rate, output_duration = metrics(args.output)
    result = {
        "target": args.target,
        "output": args.output,
        "target_sample_rate": target_rate,
        "output_sample_rate": output_rate,
        "target_duration_sec": round(target_duration, 4),
        "output_duration_sec": round(output_duration, 4),
        "duration_ratio": round(output_duration / target_duration, 4),
        "f0_contour_correlation": contour_correlation(target_f0, output_f0),
        "validation": "exploratory_svc_pitch_duration_only_not_lyric_or_speaker_gate",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
