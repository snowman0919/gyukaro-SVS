#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

rows = [json.loads(x) for x in Path("data/manifests/real_recordings.jsonl").read_text().splitlines()]
assert len(rows) == 132, len(rows)
assert [r["source_index"] for r in rows] == list(range(106,238))
assert all(Path(r["pcm_master"]).exists() for r in rows if not r["corrupt"])
assert all(r["sample_rate"] == 48000 and r["channels"] == 1 for r in rows)
print(f"PASS recordings=132 sequential=106..237 pcm=48k_mono corrupt={sum(r['corrupt'] for r in rows)}")
