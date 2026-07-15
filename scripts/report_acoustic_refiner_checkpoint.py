#!/usr/bin/env python3
"""Rebuild a concise training report from an acoustic-refiner checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = saved["training"] | {
        "stage": saved["stage"], "checkpoint": str(args.checkpoint),
        "total_parameters": sum(value.numel() for value in saved["model"].values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "report": str(args.report)}))


if __name__ == "__main__":
    main()
