#!/usr/bin/env python3
"""Measure pseudo-singing coverage gaps before any new generation."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main() -> None:
    real = read("data/manifests/real_score_accepted.jsonl"); pseudo = read("data/manifests/pseudo_singing_accepted.jsonl")
    pitches = [note["pitch"] for row in real for note in row["notes"]]; durations = [note["duration"] for row in real for note in row["notes"]]
    report = {"real_rows": len(real), "accepted_pseudo_rows": len(pseudo), "real_pitch_range": [min(pitches), max(pitches)], "real_duration_range": [min(durations), max(durations)], "real_script_shapes": dict(Counter(row["script_shape_prior"] for row in real)), "pseudo_languages": dict(Counter(row["language"] for row in pseudo)), "coverage_targets": {"new_candidates": "200-500 after gap review", "languages": ["ko", "en", "ja"], "quality_gates": ["RMVPE agreement", "duration ratio", "speaker similarity", "ASR", "language ID", "audio quality", "degeneration"]}, "status": "analysis_only_no_blind_generation"}
    Path("artifacts/reports/pseudo_singing_gap_analysis.json").write_text(json.dumps(report, indent=2) + "\n"); Path("docs/pseudo_singing_v0.5.md").write_text("# Pseudo-singing coverage (v0.5)\n\n" + json.dumps(report, indent=2) + "\n\nAccepted pseudo-singing remains low-trust (`trust_weight: 0.2`). New candidates are not admitted without the listed gates.")
    print(report)


if __name__ == "__main__": main()
