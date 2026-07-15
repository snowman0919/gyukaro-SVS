#!/usr/bin/env python3
"""Apply an experimental spectral refiner to a fixed stress manifest."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_spectral_refiner import SpectralRuntime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/reports/rc6_backend_candidate/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/acoustic_refiner_spectral_singing.pt"))
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    listening = args.output / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    refiner = SpectralRuntime(args.checkpoint)
    files, timings = {}, []
    for case, row in source["files"].items():
        audio, rate = sf.read(row["path"], dtype="float32", always_2d=True)
        mono = audio.mean(1)
        if rate != 48000:
            raise ValueError(f"candidate input must be 48 kHz: {row['path']}")
        started = time.perf_counter()
        refined = refiner.process(mono)
        timings.append(time.perf_counter() - started)
        output = mono + args.strength * (refined - mono)
        peak = float(np.max(np.abs(output)))
        safety_gain = min(1.0, 0.97 / max(peak, 1e-8))
        output *= safety_gain
        path = listening / f"{case}.wav"
        sf.write(path, output, 48000, subtype="PCM_24")
        files[case] = row | {
            "path": str(path), "source_rc6_path": row["path"],
            "spectral_refiner": str(args.checkpoint), "refiner_stage": refiner.mode,
            "refiner_strength": args.strength, "safety_gain": safety_gain,
        }
        print(case, flush=True)
    report = {
        "status": "objective_evaluation_pending",
        "name": "post-RC6 spectral-refiner probe; not a tag or release",
        "source_baseline": str(args.source), "checkpoint": str(args.checkpoint),
        "strength": args.strength,
        "selection_basis": "held-out speaker-disjoint real-degradation evaluation",
        "mean_refiner_seconds": round(float(np.mean(timings)), 4),
        "files": files, "human_review": "not_requested_until_objective_pass",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "strength", "mean_refiner_seconds")}, indent=2))


if __name__ == "__main__":
    main()
