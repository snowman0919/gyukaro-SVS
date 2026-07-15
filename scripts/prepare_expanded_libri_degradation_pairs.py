#!/usr/bin/env python3
"""Use every accepted LibriTTS-R clip except one reference per speaker."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/manifests/acoustic_clean.jsonl"
TARGET = ROOT / "data/external/manifests/libri_degradation_plan_v2.jsonl"
REPORT = ROOT / "artifacts/reports/libri_degradation_plan_v2.json"


def main() -> None:
    rows = [
        json.loads(line) for line in SOURCE.read_text().splitlines() if line
    ]
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] == "libritts_r" and row["accepted"]:
            grouped[row["speaker"]].append(row)
    plans = []
    for speaker, members in sorted(grouped.items()):
        members.sort(key=lambda row: row["id"])
        reference = members[0]
        for source in members[1:]:
            plans.append({
                "id": f"pair_v2_{source['id']}", "dataset": "libritts_r", "domain": "speech",
                "language": "en", "speaker": speaker, "source_id": source["id"],
                "clean_target": source["audio"], "reference": reference["audio"],
                "split": source["split"], "trust_weight": source["trust_weight"],
                "identity_adapter": False, "style_adapter": False,
                "license": "CC-BY-4.0", "text": source["text"],
                "asr_text_similarity": source["asr_text_similarity"],
            })
    TARGET.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans))
    report = {
        "status": "ready", "rows": len(plans), "speakers": len(grouped),
        "splits": Counter(row["split"] for row in plans), "license": "CC-BY-4.0",
        "speaker_disjoint": all(
            len({row["split"] for row in plans if row["speaker"] == speaker}) == 1
            for speaker in grouped
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
