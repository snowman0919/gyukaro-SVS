#!/usr/bin/env python3
"""Select latent-style scales on the designated quality_ko validation phrase."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import soundfile as sf

from evaluate_v07_style_semantics import proxies
from gyu_singer.inference.v07 import GyuSingerV07Renderer


GRID = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)
TARGET = {
    "soft": ("rms", -1.0),
    "breathy": ("high_frequency_ratio_4khz", 1.0),
    "energetic": ("rms", 1.0),
    "dark": ("spectral_centroid_hz", -1.0),
    "bright": ("spectral_centroid_hz", 1.0),
}


def main() -> None:
    root = Path("artifacts/reports/v07_style_scale_search")
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path("examples/quality_ko.json").read_text())
    renderer = GyuSingerV07Renderer("data/processed/master/216.wav")
    results = {}
    try:
        neutral = copy.deepcopy(base); neutral.setdefault("style", {})["preset"] = "neutral"
        neutral_path = root / "neutral.wav"
        sf.write(neutral_path, renderer.render(neutral), renderer.sample_rate, subtype="PCM_16")
        neutral_proxy = proxies(neutral_path)
        for preset, (metric, sign) in TARGET.items():
            candidates = []
            for scale in GRID:
                renderer.STYLE_CALIBRATION[preset] = scale
                score = copy.deepcopy(base); score.setdefault("style", {})["preset"] = preset
                path = root / f"{preset}_{scale:+g}.wav"
                sf.write(path, renderer.render(score), renderer.sample_rate, subtype="PCM_16")
                measured = proxies(path)
                delta = measured[metric] - neutral_proxy[metric]
                candidates.append({"scale": scale, "metric": metric, "delta": round(delta, 8), "direction_score": round(sign * delta, 8), "path": str(path)})
            results[preset] = {"selected": max(candidates, key=lambda row: row["direction_score"]), "candidates": candidates}
    finally:
        renderer.close()
    report = {"selection_set": "examples/quality_ko.json", "test_phrases_used": False, "grid": GRID, "neutral": neutral_proxy, "results": results, "selected_scales": {preset: row["selected"]["scale"] for preset, row in results.items()}}
    Path("artifacts/reports/v07_style_scale_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
