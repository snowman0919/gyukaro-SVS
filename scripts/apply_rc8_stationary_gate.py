#!/usr/bin/env python3
"""Blend RC7 0.5 and full correction only on stable score-timed vowels."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gyu_singer.inference.spectral_gate import stationary_gate  # noqa: E402


def blend(low_path: Path, high_path: Path, score_path: Path, output: Path) -> dict:
    low, low_rate = sf.read(low_path, dtype="float32", always_2d=True)
    high, high_rate = sf.read(high_path, dtype="float32", always_2d=True)
    if low_rate != 48_000 or high_rate != low_rate or low.shape != high.shape:
        raise ValueError("stationary-gate inputs must be aligned 48 kHz audio")
    gate = stationary_gate(json.loads(score_path.read_text()), len(low))[:, None]
    audio = low + gate * (high - low)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, low_rate, subtype="PCM_24")
    return {"path": str(output.relative_to(ROOT)), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "stationary_gate_mean": round(float(gate.mean()), 4)}


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_stationary_gate"
    low = json.loads((ROOT / "artifacts/reports/spectral_refiner_stress_s050/manifest.json").read_text())
    high = json.loads((ROOT / "artifacts/reports/spectral_refiner_stress_s100/manifest.json").read_text())
    stress = {}
    for case, item in low["files"].items():
        stress[case] = item | blend(ROOT / item["path"], ROOT / high["files"][case]["path"], ROOT / item["score"], root / "stress/listening" / f"{case}.wav")
    sustained_manifest = json.loads((ROOT / "artifacts/reports/rc8_sustained_set/manifest.json").read_text())
    indexed = {(row["case"], row["spectral_strength"]): row for row in sustained_manifest["rows"]}
    sustained = {}
    for case in sorted({row["case"] for row in sustained_manifest["rows"]}):
        a, b = indexed[(case, .5)], indexed[(case, 1.0)]
        sustained[case] = blend(ROOT / a["path"], ROOT / b["path"], ROOT / a["score"], root / "sustained/listening" / f"{case}.wav") | {"score": a["score"]}
    report = {
        "status": "objective_evaluation_pending", "base_strength": .5, "stationary_strength": 1.0,
        "minimum_stationary_seconds": .3, "boundary_exclusion_seconds": .08,
        "stress": stress, "sustained": sustained,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (root / "stress/manifest.json").write_text(json.dumps({"status": report["status"], "files": stress}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "stress": len(stress), "sustained": len(sustained)}, indent=2))


if __name__ == "__main__":
    main()
