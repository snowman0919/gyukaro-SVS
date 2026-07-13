#!/usr/bin/env python3
"""Export ASR-confirmed real GYU singing takes for MOSS Nano SFT."""
from __future__ import annotations
import json
from pathlib import Path

rows = [json.loads(x) for x in Path("data/manifests/real_segments.jsonl").read_text().splitlines()]
train = []
for row in rows:
    if row["script_block"] not in {"D", "E"} or row["alignment_confidence"] < .9:
        continue
    train.append({"audio": str(Path(row["source_file"]).with_suffix(".wav")), "text": row["text"], "language": "ko", "source_index": row["source_index"]})
out = Path("data/manifests/moss_sft_raw.jsonl")
# Paths are relative to this manifest; masters live in the sibling processed directory, so use absolute paths.
for row in train:
    row["audio"] = str((Path("data/processed/master") / f"{row['source_index']}.wav").resolve())
out.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in train))
print(f"moss_sft_records={len(train)}")
