#!/usr/bin/env python3
"""Adapted v1 checkpoint: selects stable real GYU voiced cycles, no fake neural training."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample

rows = [json.loads(x) for x in Path("data/manifests/real_recordings.jsonl").read_text().splitlines()]
candidates = [r for r in rows if 65 <= r["f0_median_hz"] <= 800 and r["voiced_frame_ratio"] > .30 and not r["clipping"]]
selected, used = [], []
for target in np.linspace(100, 420, 8):
    r = min((x for x in candidates if x["source_index"] not in used), key=lambda x: abs(x["f0_median_hz"]-target))
    audio, rate = sf.read(r["pcm_master"], dtype="float32")
    period = max(32, int(rate / r["f0_median_hz"]))
    frame = period * 8
    best, where = -1., 0
    for start in range(0, max(1, len(audio)-frame), period):
        power = float(np.mean(audio[start:start+frame]**2))
        if power > best: best, where = power, start
    loop = audio[where:where+frame]
    if len(loop) < frame: loop = np.pad(loop, (0, frame-len(loop)))
    # One-period crossfade makes sustained notes stable at loop boundary.
    loop[-period:] = loop[-period:] * np.linspace(1,0,period) + loop[:period] * np.linspace(0,1,period)
    selected.append(resample(loop, 2048).astype(np.float32)); used.append(r["source_index"])
np.savez_compressed("checkpoints/gyu_v1_experimental.npz", loops=np.stack(selected), f0=np.array([next(x for x in candidates if x["source_index"]==i)["f0_median_hz"] for i in used]), sample_rate=48000, source_indices=np.array(used))
Path("artifacts/reports/training.json").write_text(json.dumps({"status":"completed","kind":"real-anchor source-loop adaptation","trainable_parameters":0,"source_indices":used,"reason":"29 minutes with unverified text is insufficient for credible neural multilingual SVS from scratch"},indent=2))
print("checkpoint=checkpoints/gyu_v1_experimental.npz", "sources=", used)
