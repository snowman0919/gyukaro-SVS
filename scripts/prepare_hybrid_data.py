#!/usr/bin/env python3
"""Create reproducible phrase-score manifests for hybrid training.

Real recordings use RMVPE/script-prior reconstructed scores. They remain
explicitly inferred and are never reported as source annotation.
"""
from __future__ import annotations

import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    root = Path("data/manifests")
    real = read(root / "neural_supervision.jsonl")
    reconstructed = {row["id"]: row for row in read(root / "reconstructed_real_scores.jsonl")}
    phoneme_alignment = {row["id"]: row["phones"] for row in read(root / "real_phoneme_alignment.jsonl")}
    rows = []
    for row in real:
        score = reconstructed[row["id"]]
        rows.append({"id": row["id"], "phase": "C_real_gyu", "audio_path": row["audio_path"], "f0_path": f"data/cache/hybrid_f0/{row['id']}.npy", "language": row["language"], "text": row["text"],
                     "trust_weight": 1.0, "split": row["split"], "phoneme_alignment": phoneme_alignment[row["id"]], "score": {"language": score["language"], "tempo": 120, "sample_rate": 48000, "score_source": score["score_source"], "script_shape_prior": score["script_shape_prior"], "notes": score["notes"]}})
    accepted_path = root / "pseudo_singing_accepted.jsonl"
    if accepted_path.exists():
        for row in read(accepted_path):
            if row.get("quality_status") != "accepted" or row.get("training_license") != "allowed":
                continue
            score = row.get("score")
            if score:
                rows.append({"id": row["id"], "phase": "B_pseudo_singing", "audio_path": row["output_path"], "f0_path": f"data/cache/hybrid_f0/{row['id']}.npy", "language": row["language"], "text": row["text"],
                             "trust_weight": float(row["trust_weight"]), "split": "train", "score": score})
    train_rows = [row for row in rows if row["split"] == "train"]
    # Original corpus has no validation split. Deterministic holdout prevents silent test tuning.
    validation_ids = {row["id"] for row in train_rows[::13]}
    for row in rows:
        if row["id"] in validation_ids:
            row["split"] = "valid"
    for split in ("train", "valid", "test"):
        write(root / f"hybrid_{split}.jsonl", [row for row in rows if row["split"] == split])
    write(root / "hybrid_all.jsonl", rows)
    print({"real": len(real), "accepted_pseudo": len(rows) - len(real), "splits": {key: sum(row["split"] == key for row in rows) for key in ("train", "valid", "test")}})


if __name__ == "__main__":
    main()
