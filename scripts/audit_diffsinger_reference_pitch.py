#!/usr/bin/env python3
"""Audit an octave-ambiguous singing reference from waveform evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cents(left: float, right: float) -> float:
    return float(abs(1200 * np.log2(left / right)))


def harmonic_levels(audio: np.ndarray, rate: int, base: float, count: int = 8) -> list[float]:
    start, end = round(len(audio) * .1), round(len(audio) * .9)
    window = audio[start:end] * np.hanning(end - start)
    fft_size = 1
    while fft_size < len(window) * 4:
        fft_size *= 2
    magnitude = np.abs(np.fft.rfft(window, fft_size))
    frequency = np.fft.rfftfreq(fft_size, 1 / rate)
    values = []
    for harmonic in range(1, count + 1):
        target = base * harmonic
        center = int(np.argmin(np.abs(frequency - target)))
        radius = max(1, round(4 / (rate / fft_size)))
        values.append(float(np.max(magnitude[max(0, center - radius):center + radius + 1])))
    normalizer = max(values) or 1
    return [round(value / normalizer, 4) for value in values]


def analyze(path: Path, lower_hz: float, upper_hz: float) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    yin = librosa.yin(
        mono, fmin=100, fmax=800, sr=rate, frame_length=4096,
        hop_length=max(1, round(rate * .01)),
    )
    yin = yin[np.isfinite(yin)]
    lower = harmonic_levels(mono, rate, lower_hz)
    upper = harmonic_levels(mono, rate, upper_hz)
    median = float(np.median(yin))
    # Odd multiples of the lower candidate are not harmonics of its octave-up candidate.
    lower_odd_support = float(sum(lower[index - 1] for index in (1, 3, 5, 7)))
    lower_supported = cents(median, lower_hz) <= 100 and lower_odd_support >= .5
    return {
        "status": "lower_octave_supported" if lower_supported else "octave_unresolved",
        "audio": {
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha256(path),
            "sample_rate": rate,
            "channels": int(audio.shape[1]),
            "duration_seconds": round(len(mono) / rate, 6),
            "local_evaluation_only": True,
        },
        "yin": {
            "p05_hz": round(float(np.percentile(yin, 5)), 2),
            "median_hz": round(median, 2),
            "p95_hz": round(float(np.percentile(yin, 95)), 2),
            "lower_candidate_error_cents": round(cents(median, lower_hz), 2),
            "upper_candidate_error_cents": round(cents(median, upper_hz), 2),
        },
        "candidates": {
            "lower": {"hz": lower_hz, "normalized_harmonics_1_to_8": lower},
            "upper": {"hz": upper_hz, "normalized_harmonics_1_to_8": upper},
            "lower_odd_harmonic_support": round(lower_odd_support, 4),
        },
        "interpretation": (
            "YIN supports the lower candidate and energy at its odd harmonics shows that the "
            "upper spectral peak is a second harmonic, not the fundamental."
            if lower_supported else
            "Waveform evidence is insufficient to resolve the octave automatically."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--lower-hz", type=float, default=261.625565)
    parser.add_argument("--upper-hz", type=float, default=523.251131)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.audio.resolve(), args.lower_hz, args.upper_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
