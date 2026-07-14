#!/usr/bin/env python3
"""Evaluate predicted expressive F0 against held-out real GYU RMVPE targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gyu_singer.data import acoustic_reference_features
from gyu_singer.inference.quality_controller import QualityPitchController


def main() -> None:
    rows = [json.loads(line) for line in Path("data/manifests/real_gyu_prosody.jsonl").read_text().splitlines() if line and json.loads(line).get("split") == "test"]
    controller = QualityPitchController("checkpoints/gyu_prosody_v0.5.pt", acoustic_reference_features("data/processed/master/216.wav"))
    records = []
    for row in rows:
        residual, duration = controller.predict(row["score"]); residual = residual.cpu().numpy(); f0 = np.load(row["target_f0_path"]); nominal = []
        notes = row["score"]["notes"]; frames = len(f0); times = np.arange(frames, dtype=np.float32) / 12.5
        for time in times:
            note = next((note for note in notes if note["start"] <= time < note["start"] + note["duration"]), notes[-1]); nominal.append(440 * 2 ** ((note["pitch"] - 69) / 12))
        nominal = np.asarray(nominal); expressive = nominal * 2 ** (np.interp(times, np.linspace(0, duration, len(residual)), residual) / 12)
        voiced = f0 > 1; if_any = int(voiced.sum()) > 2
        corr = float(np.corrcoef(expressive[voiced], f0[voiced])[0, 1]) if if_any else 0.0; mae = float(np.median(np.abs(1200 * np.log2(np.maximum(expressive[voiced], 1) / np.maximum(f0[voiced], 1))))) if if_any else 1200.0
        nominal_mae = float(np.median(np.abs(1200 * np.log2(np.maximum(nominal[voiced], 1) / np.maximum(f0[voiced], 1))))) if if_any else 1200.0
        records.append({"id": row["id"], "correlation": round(corr, 4), "pitch_mae_cents": round(mae, 2), "nominal_pitch_mae_cents": round(nominal_mae, 2), "improvement_cents": round(nominal_mae - mae, 2)})
    report = {"checkpoint": "checkpoints/gyu_prosody_v0.5.pt", "heldout_rows": len(records), "target": "real GYU RMVPE F0", "metrics": {"correlation_mean": round(float(np.mean([r["correlation"] for r in records])), 4), "pitch_mae_cents_median": round(float(np.median([r["pitch_mae_cents"] for r in records])), 2), "nominal_mae_cents_median": round(float(np.median([r["nominal_pitch_mae_cents"] for r in records])), 2), "improvement_cents_median": round(float(np.median([r["improvement_cents"] for r in records])), 2)}, "rows": records}
    Path("artifacts/reports/evaluation_v0.5_prosody.json").write_text(json.dumps(report, indent=2) + "\n"); Path("docs/evaluation_v0.5.md").write_text("# v0.5 evaluation\n\n" + json.dumps(report, indent=2) + "\n")
    print(report["metrics"])


if __name__ == "__main__": main()
