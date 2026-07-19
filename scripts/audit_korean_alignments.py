#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter

from gyu_singer.evaluation.korean_lexical import classify_alignment


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    rows = []
    for source, weight in (
        ("data/manifests/real_phoneme_alignment_all.jsonl", 0.4),
        ("data/manifests/diffsinger_ko_phoneme_prior.jsonl", 0.25),
    ):
        for row in read_jsonl(ROOT / source):
            confidence = row.get("confidence")
            if confidence is None and row.get("ctc_mean_log_score") is not None:
                confidence = math.exp(float(row["ctc_mean_log_score"]))
            confidence = float(confidence or 0.0)
            classification = classify_alignment("inferred", confidence)
            rows.append({
                "id": row["id"], "source_manifest": source,
                "alignment_source": row.get("alignment_source", "MMS_CTC_inferred_timing"),
                "label_status": row.get("label_status", "inferred"),
                "confidence": round(confidence, 6), "classification": classification,
                "training_weight": weight,
                "use_policy": "linguistic_prior_only_if_representation_selected",
            })
    manifest = ROOT / "data/manifests/korean_alignment_audit.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    counts = Counter(row["classification"] for row in rows)
    report = {
        "status": "alignment_audit_complete_training_blocked",
        "rows": len(rows), "classification_counts": dict(sorted(counts.items())),
        "manual_rows": counts["manual"], "inferred_rows": counts["inferred_only"],
        "training_started": False,
    }
    destination = ROOT / "artifacts/reports/korean_phone_reassessment/alignment_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
