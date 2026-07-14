#!/usr/bin/env python3
"""Build non-duplicated Fish/MOSS/Higgs paired teacher manifest.

Rows are grouped by benchmark id; no synthetic duplication is performed.
Splits are assigned by benchmark id so text/reference combinations cannot leak.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path


SOURCE = Path("data/manifests/teacher_weighted.jsonl")
DEST = Path("data/manifests/teacher_internal_pairs.jsonl")


def main() -> None:
    grouped: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for line in SOURCE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped[row["id"]][row["teacher"]] = row

    rows = []
    for benchmark_id, teachers in sorted(grouped.items()):
        if not {"fish_s2_pro", "moss_local_v15"}.issubset(teachers):
            continue
        fish, moss = teachers["fish_s2_pro"], teachers["moss_local_v15"]
        higgs = teachers.get("higgs_tts3_4b")
        trust = {"fish_s2_pro": float(fish.get("trust_weight", 0.0)),
                 "moss_local_v15": float(moss.get("trust_weight", 0.0)),
                 "higgs_tts3_4b": float(higgs.get("trust_weight", 0.0)) if higgs else 0.0}
        agreements = [float(r.get("teacher_agreement_score", 0.0)) for r in (fish, moss, higgs) if r]
        # Agreement is evidence weighting only; it is never used to fabricate labels.
        row = {
            "benchmark_id": benchmark_id,
            "language": fish["language"],
            "text": fish["text"],
            "reference_ids": fish.get("reference_ids", []),
            "fish_output": fish["output_path"],
            "moss_output": moss["output_path"],
            "higgs_output": higgs.get("output_path") if higgs else None,
            "fish_trust": trust["fish_s2_pro"],
            "moss_trust": trust["moss_local_v15"],
            "higgs_trust": trust["higgs_tts3_4b"],
            "cross_teacher_agreement": sum(agreements) / len(agreements),
            "teacher_count": len(agreements),
            "split": "train",
        }
        rows.append(row)
    # Stratify deterministic held-out rows by language; benchmark IDs remain disjoint.
    for language in sorted({r["language"] for r in rows}):
        language_rows = [r for r in rows if r["language"] == language]
        for index, row in enumerate(language_rows):
            row["split"] = "test" if index == 0 else ("validation" if index == 1 else "train")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print({"rows": len(rows), "languages": dict(collections.Counter(r["language"] for r in rows)),
           "splits": dict(collections.Counter(r["split"] for r in rows)),
           "with_higgs": sum(r["higgs_output"] is not None for r in rows)})


if __name__ == "__main__":
    main()
