#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf

rows = [json.loads(x) for x in Path("data/manifests/teacher_pilot.jsonl").read_text().splitlines()]
accepted = []
for row in rows:
    audio, rate = sf.read(row["output_path"], dtype="float32")
    peak, rms = float(np.max(np.abs(audio))), float(np.sqrt(np.mean(audio**2)))
    silence = float(np.mean(np.abs(audio) < .005))
    row.update({"duration_sec":round(len(audio)/rate,3),"peak":round(peak,5),"rms":round(rms,5),"silence_ratio":round(silence,4),"speaker_score":None,"content_score":None,"language_score":None,"teacher_agreement_score":None,"overall_confidence":None,"quality_status":"acoustic_pass_only" if peak < .995 and rms > .001 else "rejected_acoustic"})
    accepted.append(row)
Path("data/manifests/teacher_filtered.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in accepted))
assert all(x["quality_status"] != "rejected_acoustic" for x in accepted)
print(f"acoustic_pass={len(accepted)}; ASR/speaker gates pending")
