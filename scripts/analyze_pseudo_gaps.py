#!/usr/bin/env python3
"""Report real-score versus accepted pseudo-singing coverage without relabeling inference as truth."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: str) -> list[dict]:
    candidate = Path(path)
    return [json.loads(line) for line in candidate.read_text().splitlines() if line] if candidate.exists() else []


def coverage(rows: list[dict]) -> dict:
    notes = [note for row in rows for note in row.get("notes", row.get("score", {}).get("notes", []))]
    ordered = [sorted(row.get("notes", row.get("score", {}).get("notes", [])), key=lambda note: note.get("start", 0)) for row in rows]
    pitches = [note["pitch"] for note in notes if "pitch" in note]
    durations = [note["duration"] for note in notes if "duration" in note]
    intervals = [b["pitch"] - a["pitch"] for sequence in ordered for a, b in zip(sequence, sequence[1:]) if "pitch" in a and "pitch" in b]
    onset_gaps = [b.get("start", 0) - a.get("start", 0) for sequence in ordered for a, b in zip(sequence, sequence[1:])]
    return {"rows": len(rows), "notes": len(notes), "languages": dict(Counter(row.get("language", "unknown") for row in rows)), "pitch_range_midi": [min(pitches), max(pitches)] if pitches else None, "interval_abs_midi": {"mean": round(sum(map(abs, intervals)) / len(intervals), 3) if intervals else None, "large_jump_count_ge5": sum(abs(value) >= 5 for value in intervals)}, "duration_sec": {"min": round(min(durations), 3) if durations else None, "max": round(max(durations), 3) if durations else None, "sustained_count_ge0_8": sum(value >= .8 for value in durations)}, "transitions": {"repeated_note_count": sum(value == 0 for value in intervals), "ascending_count": sum(value > 0 for value in intervals), "descending_count": sum(value < 0 for value in intervals)}, "tempo_proxy_bpm_from_note_onsets": round(60 / (sum(onset_gaps) / len(onset_gaps)), 2) if onset_gaps and sum(onset_gaps) > 0 else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="data/manifests/real_score_accepted.jsonl")
    parser.add_argument("--pseudo", default="data/manifests/pseudo_singing_v06_accepted.jsonl")
    args = parser.parse_args()
    report = {"real_gyu_score_corpus": coverage(read(args.real)), "accepted_pseudo_singing": coverage(read(args.pseudo)), "targeted_generation": {"candidate_target": "200-500", "coverage": ["pitch range", "interval", "duration", "tempo proxy", "sustained vowels", "repeated notes", "ascending", "descending", "large jumps", "language"], "gate": ["RMVPE agreement", "duration ratio", "speaker similarity", "ASR", "language ID", "audio quality", "degeneration"]}, "label_policy": "Pseudo note features are RMVPE-inferred and remain low-trust; they are not real-GYU prosody labels."}
    Path("artifacts/reports/pseudo_singing_v06_gap_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/pseudo_singing_v0.6.md").write_text("# v0.6 pseudo-singing coverage\n\n" + json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
