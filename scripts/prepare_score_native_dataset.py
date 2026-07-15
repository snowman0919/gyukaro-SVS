#!/usr/bin/env python3
"""Build the non-destructive score-native acoustic training manifest."""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data/manifests" / name).read_text().splitlines() if line]


def main() -> None:
    scores = {row["id"]: row for row in rows("real_score_accepted.jsonl")}
    alignments = {row["id"]: row for row in rows("real_phoneme_alignment.jsonl")}
    verified = {row["id"] for row in rows("manual_verified_scores.jsonl")}
    output = []
    for item_id in sorted(scores.keys() & alignments.keys()):
        score, alignment = scores[item_id], alignments[item_id]
        audio = ROOT / score["source_audio_path"]
        if not audio.is_file():
            continue
        phones = alignment["phones"]
        output.append({
            "id": item_id,
            "audio_path": str(audio.relative_to(ROOT)),
            "duration_seconds": round(sf.info(audio).duration, 6),
            "language": score["language"],
            "ph_seq": [phone["symbol"] for phone in phones],
            "ph_dur": [round(float(phone["duration"]), 6) for phone in phones],
            "f0_path": score["f0_path"],
            "score_notes": score["notes"],
            "score_label_status": score["label_status"],
            "alignment_source": alignment["alignment_source"],
            "split": "independent_evaluation" if item_id in verified else score["split"],
            "training_allowed": item_id not in verified,
        })
    target = ROOT / "data/manifests/diffsinger_score_native.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    by_split = {}
    for row in output:
        bucket = by_split.setdefault(row["split"], {"rows": 0, "minutes": 0.0})
        bucket["rows"] += 1
        bucket["minutes"] += row["duration_seconds"] / 60
    report = {
        "status": "manifest_ready_training_not_authorized_by_data_volume",
        "rows": len(output),
        "training_rows": sum(row["training_allowed"] for row in output),
        "training_minutes": round(sum(row["duration_seconds"] for row in output if row["training_allowed"]) / 60, 3),
        "independent_evaluation_rows": sum(not row["training_allowed"] for row in output),
        "by_split": {key: {"rows": val["rows"], "minutes": round(val["minutes"], 3)} for key, val in by_split.items()},
        "target_f0_used_as_prosody_input": False,
        "real_f0_used_as_acoustic_condition": True,
        "raw_audio_copied": False,
        "warning": "Insufficient duration for production-quality acoustic training from scratch; do not package a checkpoint trained only on this manifest.",
    }
    (ROOT / "artifacts/reports/score_native_data_readiness.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
