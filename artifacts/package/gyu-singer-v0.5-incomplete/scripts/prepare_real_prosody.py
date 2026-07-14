#!/usr/bin/env python3
"""Build score-only conditioning rows with real GYU F0 as target only."""
from __future__ import annotations

import json
from pathlib import Path


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main() -> None:
    accepted = read("data/manifests/real_score_accepted.jsonl")
    rows = []
    for row in accepted:
        rows.append({
            "id": row["id"],
            "language": row["language"],
            "text": row["text"],
            "score": {"language": row["language"], "tempo": 120, "sample_rate": 48000, "score_source": row["score_source"], "notes": row["notes"]},
            "audio_path": row["source_audio_path"],
            "target_f0_path": row["f0_path"],
            "target": "real_gyu_rmvpe_log_f0",
            "target_voiced_only": True,
            "condition_inputs": ["nominal_score_f0", "score_pitch_curve", "voiced_mask", "phoneme_alignment", "style_metadata"],
            "forbidden_condition_inputs": ["target_f0_path", "real_log_f0", "actual_rmvpe_f0"],
            "alignment_source": "real_phoneme_alignment.jsonl",
            "confidence": row["quality"]["confidence"],
            "quality_flags": row["quality"]["flags"],
            "split": row.get("split", "train"),
            "trust_weight": 1.0,
        })
    output = Path("data/manifests/real_gyu_prosody.jsonl")
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {"rows": len(rows), "target": "real_gyu_rmvpe_log_f0", "synthetic_rows_primary": False, "target_f0_in_condition": False, "train": sum(row["split"] == "train" for row in rows), "validation": sum(row["split"] == "validation" for row in rows), "test": sum(row["split"] == "test" for row in rows)}
    Path("artifacts/reports/real_gyu_prosody_dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
