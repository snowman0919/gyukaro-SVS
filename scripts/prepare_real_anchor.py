#!/usr/bin/env python3
"""Create conservative anchor segments; source-order labels remain unverified."""
from __future__ import annotations
import json
from pathlib import Path
import soundfile as sf

rows = [json.loads(x) for x in Path("data/manifests/real_recordings.jsonl").read_text().splitlines()]
anchors = []
for n, row in enumerate(rows):
    if row["duration_sec"] < 1.0 or row["silence_ratio"] > .8 or row["clipping"]: continue
    audio, rate = sf.read(row["pcm_master"], dtype="float32")
    active = abs(audio) > .015
    start, end = next((i for i,v in enumerate(active) if v), 0), len(audio)
    while end > start and not active[end-1]: end -= 1
    if (end-start)/rate < 1: continue
    anchors.append({"id": row["id"], "audio_path": row["pcm_master"], "start_sec": round(start/rate,3), "end_sec": round(end/rate,3), "language":"ko", "text":"", "text_status":"unverified", "script_block":row["script_block"], "quality_score":round(min(1, row["voiced_frame_ratio"]+.25),3), "trust_weight":1.0, "split":"test" if n % 17 == 0 else "validation" if n % 11 == 0 else "train"})
Path("data/manifests/real_segments.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in anchors))
Path("data/manifests/script_alignment.jsonl").write_text("".join(json.dumps({"source_index":r["source_index"],"script_block":r["script_block"],"candidate":"source-order block assignment only","confidence":.35,"status":"needs human transcript review"},ensure_ascii=False)+"\n" for r in rows))
print(f"anchors={len(anchors)}")
