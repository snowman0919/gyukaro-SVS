#!/usr/bin/env python3
"""Render the RC8 sustained-noise diagnostic set through frozen RC7 policy."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_spectral_refiner import SpectralRuntime  # noqa: E402
from gyu_singer.inference.rc6 import GyuSingerRC6Renderer  # noqa: E402


CASES = {
    "low_soft_straight": (52, "soft", False, .45),
    "mid_neutral_straight": (64, "neutral", False, .8),
    "high_energetic_straight": (72, "energetic", False, 1.2),
    "low_neutral_vibrato": (55, "neutral", True, .8),
    "mid_soft_vibrato": (64, "soft", True, .45),
    "high_energetic_vibrato": (72, "energetic", True, 1.2),
}


class FrozenRC7Generation(GyuSingerRC6Renderer):
    @classmethod
    def _content_warp_strength(cls, score: dict) -> float:
        if cls._rapid(score):
            return 1.0
        return .25 if score["language"] == "en" else 0.0


def score(pitch: int, preset: str, vibrato: bool, dynamics: float) -> dict:
    pitch_curve = []
    if vibrato:
        pitch_curve = [{"time": round(t, 3), "value": round(.28 * math.sin(2 * math.pi * 5.5 * t), 4)} for t in np.arange(.8, 4.81, .04)]
    return {
        "language": "ko", "tempo": 60, "sample_rate": 48_000,
        "notes": [{"pitch": pitch, "start": 0, "duration": 5, "lyric": "아"}],
        "curves": {"pitch": pitch_curve, "dynamics": [{"time": 0, "value": dynamics}], "vibrato": [{"time": 0, "value": 1.0 if vibrato else 0.0}]},
        "style": {"preset": preset},
    }


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_sustained_set"
    scores = root / "scores"
    listening = root / "listening"
    scores.mkdir(parents=True, exist_ok=True); listening.mkdir(parents=True, exist_ok=True)
    renderer = FrozenRC7Generation(ROOT / "data/processed/master/216.wav", root=ROOT)
    refiner = SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_singing.pt")
    rows = []; started = time.perf_counter()
    try:
        for case, arguments in CASES.items():
            data = score(*arguments)
            score_path = scores / f"{case}.json"
            score_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            source = renderer.render(data)
            refined = refiner.process(source)
            for strength in (0.0, .25, .5, 1.0):
                audio = source + strength * (refined - source)
                audio *= min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))
                path = listening / f"{case}_spectral_{strength:g}.wav"
                sf.write(path, audio, 48_000, subtype="PCM_24")
                rows.append({"case": case, "score": str(score_path.relative_to(ROOT)), "spectral_strength": strength, "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print(case, flush=True)
    finally:
        renderer.close()
    report = {
        "status": "objective_evaluation_pending", "frozen_rc7_generation_policy": True,
        "coverage": {"registers": ["low", "mid", "high"], "dynamics": ["soft", "neutral", "energetic"], "vibrato": [False, True]},
        "spectral_strengths": [0, .25, .5, 1], "render_seconds": round(time.perf_counter() - started, 3), "rows": rows,
    }
    (root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
