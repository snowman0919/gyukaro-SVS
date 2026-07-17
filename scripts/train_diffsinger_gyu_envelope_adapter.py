#!/usr/bin/env python3
"""Estimate a language-independent, gain-neutral GYU mel-envelope direction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_diffsinger_gyu_mel_adapter import ROOT, predict_pairs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=float, default=.5)
    args = parser.parse_args()
    binary = ROOT / "data/external/work/diffsinger_score_native/binary_gyu_phrase_chunks"
    pairs = predict_pairs(args.model.resolve(), binary)
    train_input, train_target = pairs["train"]
    valid_input, valid_target = pairs["valid"]
    raw = np.median(train_target - train_input, axis=0)
    # Remove broadband gain; only the relative mel envelope is transferable.
    centered = raw - np.mean(raw)
    kernel = np.ones(9, dtype=np.float32) / 9
    smooth = np.convolve(np.pad(centered, (4, 4), mode="edge"), kernel, mode="valid")
    delta = np.clip(smooth, -args.limit, args.limit).astype(np.float32)
    baseline = float(np.mean(np.abs(valid_input - valid_target)))
    adapted = float(np.mean(np.abs(valid_input + delta - valid_target)))
    saved = {
        "delta": torch.from_numpy(delta),
        "config": {"bins": 128, "limit": args.limit, "gain_neutral": True},
        "input": "score-nominal-F0 foundation mel",
        "target": "real GYU phrase-chunk mel",
        "target_f0_used_as_condition": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    report = {
        "status": "trained_objective_evaluation_pending",
        "checkpoint": str(args.output),
        "checkpoint_sha256": digest(args.output),
        "parameters": 128,
        "method": "median paired residual, broadband gain removed, 9-bin smoothed and bounded",
        "delta_mean": round(float(delta.mean()), 7),
        "delta_rms": round(float(np.sqrt(np.mean(delta ** 2))), 7),
        "delta_min": round(float(delta.min()), 7),
        "delta_max": round(float(delta.max()), 7),
        "validation_l1_baseline": round(baseline, 6),
        "validation_l1_adapted": round(adapted, 6),
        "real_target_f0_condition_leakage": False,
        "independent_evaluation_song_in_training": False,
        "release_allowed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
