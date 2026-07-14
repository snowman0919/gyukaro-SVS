#!/usr/bin/env python3
"""Build non-duplicated Fish/MOSS/Higgs paired teacher manifest.

Rows are grouped by benchmark id; no synthetic duplication is performed.
Splits are assigned by benchmark id so text/reference combinations cannot leak.
"""
from __future__ import annotations

import collections
import hashlib
import json
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--output", default="data/manifests/teacher_internal_pairs.jsonl")
    args = parser.parse_args()
    grouped: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for line in Path(args.input).read_text().splitlines():
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
            "style": fish.get("style", "neutral"),
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
        semantic = json.dumps([row["language"], row["text"], row["reference_ids"]], ensure_ascii=False, separators=(",", ":"))
        row["semantic_group_id"] = hashlib.sha256(semantic.encode()).hexdigest()[:16]
        rows.append(row)
    # Keep every text/reference semantic group in one split.  Style variants are
    # real teacher outputs, but must not leak the same text/reference pair.
    for language in sorted({r["language"] for r in rows}):
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            if row["language"] == language:
                groups[row["semantic_group_id"]].append(row)
        ordered = [groups[key] for key in sorted(groups)]
        held_out = max(1, round(len(ordered) * .1))
        for index, group in enumerate(ordered):
            split = "test" if index < held_out else ("validation" if index < held_out * 2 else "train")
            for row in group:
                row["split"] = split
    split_groups = collections.defaultdict(set)
    for row in rows:
        split_groups[row["semantic_group_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in split_groups.values()), "semantic group leaked across splits"
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print({"rows": len(rows), "languages": dict(collections.Counter(r["language"] for r in rows)),
           "splits": dict(collections.Counter(r["split"] for r in rows)),
           "semantic_groups": len(split_groups),
           "with_higgs": sum(r["higgs_output"] is not None for r in rows)})


if __name__ == "__main__":
    main()
