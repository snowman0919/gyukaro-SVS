#!/usr/bin/env python3
"""Summarize measurable GYU RMVPE prosody tendencies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    rows = [json.loads(line) for line in Path("data/manifests/real_score_accepted.jsonl").read_text().splitlines() if line]; rates = []; extents = []; attacks = []; drifts = []
    for row in rows:
        f0 = np.load(row["f0_path"]); voiced = f0 > 1
        if voiced.sum() < 5: continue
        cents = 1200 * np.log2(np.maximum(f0[voiced], 1) / np.median(f0[voiced])); extents.append(float(np.percentile(cents, 95) - np.percentile(cents, 5)))
        if len(cents) > 8:
            crossings = np.sum(np.diff(np.signbit(cents - np.median(cents))) != 0); rates.append(float(crossings / 2 / (voiced.sum() / 12.5)))
        attacks.append(float(np.median(cents[: max(1, min(3, len(cents))) ]))); drifts.append(float(cents[-1] - cents[0]))
    report = {"rows": len(rows), "vibrato_rate_proxy_hz_median": round(float(np.median(rates)), 3), "vibrato_extent_proxy_cents_median": round(float(np.median(extents)), 2), "note_attack_residual_proxy_cents_median": round(float(np.median(attacks)), 2), "sustained_drift_proxy_cents_median": round(float(np.median(drifts)), 2), "method": "RMVPE voiced frames; proxies, not hand labels"}
    Path("artifacts/reports/gyu_prosody_analysis.json").write_text(json.dumps(report, indent=2) + "\n"); Path("docs/gyu_prosody_analysis.md").write_text("# GYU prosody analysis (v0.5)\n\n" + json.dumps(report, indent=2) + "\n\nAll measurements are inferred from real GYU RMVPE frames; they are supervision diagnostics, not source annotations.")
    print(report)


if __name__ == "__main__": main()
