#!/usr/bin/env python3
"""Merge verified scores with trusted reconstructed rows for v0.6 training."""
import json
from pathlib import Path

out = []
for raw in Path("data/manifests/manual_verified_scores.jsonl").read_text().splitlines():
    row = json.loads(raw); notes = row["notes"]; shift = min(float(n["start"]) for n in notes)
    verified_notes = []; cursor = 0.0
    for n in notes:
        item = {**n, "start": round(cursor, 5)}; verified_notes.append(item); cursor += float(n["duration"])
    score = {"language": row["language"], "tempo": 120, "sample_rate": 48000, "score_source": "independent_verified_score", "notes": verified_notes}
    out.append({"id": row["id"], "language": row["language"], "text": row["text"], "score": score, "audio_path": row["audio_path"], "target_f0_path": f"data/cache/hybrid_f0/{row['id']}.npy", "target": "real_gyu_rmvpe_log_f0", "confidence": 1.0, "trust_weight": 1.0, "score_label": "verified_score"})
for raw in Path("data/manifests/real_gyu_prosody.jsonl").read_text().splitlines():
    row = json.loads(raw)
    if float(row.get("confidence", 0.0)) >= 0.7:
        row["score_label"] = "high_confidence_reconstructed_score"; row["trust_weight"] = min(0.9, float(row.get("confidence", 0.7))); out.append(row)
Path("data/manifests/real_gyu_prosody_v06.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print({"rows": len(out), "verified": sum(r["score_label"] == "verified_score" for r in out), "reconstructed": sum(r["score_label"] != "verified_score" for r in out)})
