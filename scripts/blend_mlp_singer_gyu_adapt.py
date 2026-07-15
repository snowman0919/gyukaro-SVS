#!/usr/bin/env python3
"""Create bounded interpolation gates for the GYU score-native adaptation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "data/cache/mlp-singer"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MODEL_ROOT))

from train_mlp_singer_gyu_adapt import remap_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    base, config = remap_model()
    base_state = base.state_dict()
    adapted = torch.load(
        MODEL_ROOT / "checkpoints/gyu_adapt_full/steps_400.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    output = MODEL_ROOT / "checkpoints/gyu_adapt_blend"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for strength in (0.0, 0.1, 0.25, 0.5):
        state = {
            key: base_state[key] + strength * (adapted[key] - base_state[key])
            for key in base_state
        }
        name = f"strength_{round(strength * 100):03d}.pt"
        target = output / name
        torch.save({"model": state, "adaptation_strength": strength}, target)
        rows.append({"strength": strength, "checkpoint": name, "sha256": sha256(target)})
    report = {
        "status": "blend_gate_ready",
        "base": "official MLP Singer with lossless expanded pitch vocabulary",
        "adapted": "400-step full GYU adaptation",
        "training_scores": "inferred; independent verified rows excluded",
        "rows": rows,
        "production_integrated": False,
    }
    (ROOT / "artifacts/reports/mlp_singer_gyu_blend.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
