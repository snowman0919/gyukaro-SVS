#!/usr/bin/env python3
"""Create a bounded acoustic-checkpoint interpolation for diagnostic ablation.

This is not speaker identity transfer.  It is used to test whether a lower
register foundation direction can reduce the perceived high-register timbre
without destroying score-native pronunciation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be in [0, 1]")

    source_path = args.source.resolve()
    direction_path = args.direction.resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    direction = torch.load(direction_path, map_location="cpu", weights_only=False)
    source_state = source["state_dict"]
    direction_state = direction["state_dict"]
    if source_state.keys() != direction_state.keys():
        raise ValueError("checkpoint state keys differ")

    changed = 0
    for key, value in source_state.items():
        other = direction_state[key]
        if value.shape != other.shape:
            raise ValueError(f"shape mismatch for {key}: {value.shape} != {other.shape}")
        if value.is_floating_point():
            source_state[key] = torch.lerp(value, other.to(value.dtype), args.alpha)
            changed += int(not torch.equal(value, other))
        elif not torch.equal(value, other):
            raise ValueError(f"non-floating state differs for {key}")

    output = args.output_experiment.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in ("config.yaml", "dictionary-gyu.txt", "lang_map.json", "spk_map.json"):
        shutil.copy2(args.source_experiment.resolve() / name, output / name)
    source["global_step"] = 1
    checkpoint = output / "model_ckpt_steps_1.ckpt"
    torch.save(source, checkpoint)
    report = {
        "status": "diagnostic_only",
        "method": "linear full-acoustic checkpoint interpolation",
        "identity_claim": False,
        "source": str(source_path),
        "source_sha256": digest(source_path),
        "direction": str(direction_path),
        "direction_sha256": digest(direction_path),
        "alpha": args.alpha,
        "changed_float_tensors": changed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "release_allowed": False,
    }
    (output / "interpolation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
