#!/usr/bin/env python3
"""Test inferred acoustic directions for every v0.7 latent style preset."""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from gyu_singer.inference.v07 import GyuSingerV07Renderer


PRESETS = ("neutral", "soft", "breathy", "energetic", "dark", "bright")
REPORT = Path("artifacts/reports")


def proxies(path: Path) -> dict[str, float]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    frequency, power = welch(audio, rate, nperseg=min(2048, len(audio)))
    total = max(float(power.sum()), 1e-12)
    return {
        "spectral_centroid_hz": round(float((frequency * power).sum()) / total, 6),
        "rms": round(float(np.sqrt(np.mean(audio * audio))), 8),
        "high_frequency_ratio_4khz": round(float(power[frequency >= 4000].sum()) / total, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="quality,heldout,korean")
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = {}
    if args.metrics_only:
        paths = {f"{split}_{preset}": str(REPORT / f"v07_style_semantics_{split}_{preset}.wav") for split in splits for preset in PRESETS}
    else:
        renderer = GyuSingerV07Renderer("data/processed/master/216.wav")
        try:
            for split in splits:
                source = Path(f"examples/{split}_ko.json") if split in {"quality", "heldout"} else Path(f"examples/{split}.json")
                base = json.loads(source.read_text())
                for preset in PRESETS:
                    score = copy.deepcopy(base)
                    score.setdefault("style", {})["preset"] = preset
                    path = REPORT / f"v07_style_semantics_{split}_{preset}.wav"
                    sf.write(path, renderer.render(score), renderer.sample_rate, subtype="PCM_16")
                    paths[f"{split}_{preset}"] = str(path)
        finally:
            renderer.close()

    measurements = {key: proxies(Path(path)) for key, path in paths.items()}
    deltas = {}
    checks = {}
    for split in splits:
        neutral = measurements[f"{split}_neutral"]
        deltas[split] = {preset: {key: round(measurements[f"{split}_{preset}"][key] - neutral[key], 6) for key in neutral} for preset in PRESETS[1:]}
        checks[split] = {
            "soft_rms_lower": deltas[split]["soft"]["rms"] < 0,
            "breathy_high_frequency_ratio_higher": deltas[split]["breathy"]["high_frequency_ratio_4khz"] > 0,
            "energetic_rms_higher": deltas[split]["energetic"]["rms"] > 0,
            "dark_centroid_lower": deltas[split]["dark"]["spectral_centroid_hz"] < 0,
            "bright_centroid_higher": deltas[split]["bright"]["spectral_centroid_hz"] > 0,
        }
    report = {
        "protocol": "Sign calibration selected on quality_ko only. heldout_ko was observed during development; korean.json is the new locked confirmation. Same score/content/reference/F0; only latent style condition changes.",
        "semantic_evidence_is_proxy_not_listening_proof": True,
        "calibration": GyuSingerV07Renderer.STYLE_CALIBRATION,
        "semantic_status": GyuSingerV07Renderer.STYLE_SEMANTICS,
        "paths": paths,
        "measurements": measurements,
        "deltas_from_neutral": deltas,
        "direction_checks": checks,
        "all_directions_pass": {split: all(checks[split].values()) for split in splits},
        "validated_semantic_directions_pass": {split: all(checks[split][key] for key in ("breathy_high_frequency_ratio_higher", "energetic_rms_higher")) for split in splits},
    }
    Path("artifacts/reports/v07_style_semantics.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/style_adapter_v0.7.md").write_text("# v0.7 latent acoustic-style adapter\n\n" + json.dumps(report, indent=2) + "\n\nAll semantic claims above are inferred acoustic proxies. Listening validation remains separate.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
