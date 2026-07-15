#!/usr/bin/env python3
"""Apply a measured low-strength GYU refiner to the fixed RC5 stress set."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/reports/rc5_stress_candidate4/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/refiner_rc_candidate"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/acoustic_refiner_gyu.pt"))
    parser.add_argument("--strength", type=float, default=.25)
    args = parser.parse_args()
    source = json.loads(args.source.read_text()); listening = args.output / "listening"; listening.mkdir(parents=True, exist_ok=True)
    refiner = AcousticRefinerRuntime(args.checkpoint); files = {}; timings = []
    for case, row in source["files"].items():
        audio, rate = sf.read(row["path"], dtype="float32", always_2d=True); mono = audio.mean(1)
        if rate != 48000:
            raise ValueError(f"candidate input must be 48 kHz: {row['path']}")
        started = time.perf_counter(); refined = refiner.process(mono); timings.append(time.perf_counter() - started)
        output = mono + args.strength * (refined - mono); peak = float(np.max(np.abs(output))); output *= min(1.0, .97 / max(peak, 1e-8))
        path = listening / f"{case}.wav"; sf.write(path, output, 48000, subtype="PCM_24")
        files[case] = row | {"path": str(path), "source_rc5_path": row["path"], "acoustic_refiner": str(args.checkpoint), "refiner_stage": "universal+singing+GYU", "refiner_strength": args.strength}
        print(case, flush=True)
    report = {"status": "objective_evaluation_pending", "name": "post-RC5 acoustic-refiner candidate (not a tag or release)",
              "source_baseline": str(args.source), "checkpoint": str(args.checkpoint), "strength": args.strength,
              "selection_basis": "held-out real-GYU sweep: 0.25 improved log spectral/highband/envelope distance and HF spike/sample jump; human listening pending",
              "mean_refiner_seconds": round(float(np.mean(timings)), 4), "files": files, "human_review": "pending"}
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "checkpoint", "strength", "mean_refiner_seconds")}, indent=2))


if __name__ == "__main__":
    main()
