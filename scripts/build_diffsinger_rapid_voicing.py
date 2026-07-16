#!/usr/bin/env python3
"""Apply the canonical phoneme voicing mask to the exact rapid DiffSinger score."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/work/pjs/rapid_repeat.ds"
TARGET = ROOT / "data/external/work/pjs/rapid_repeat_voiced.ds"
UNVOICED = {"ja_ch", "ja_f", "ja_h", "ja_hy", "ja_k", "ja_ky", "ja_p",
            "ja_s", "ja_sh", "ja_t", "ja_ts", "SP", "AP"}


def mask_f0(phones: list[str], durations: list[float], f0: list[float], timestep: float) -> list[float]:
    boundaries = []
    elapsed = 0.0
    for phone, duration in zip(phones, durations):
        boundaries.append((elapsed, elapsed + duration, phone))
        elapsed += duration
    result = []
    for index, value in enumerate(f0):
        center = (index + 0.5) * timestep
        phone = next((symbol for start, end, symbol in boundaries if start <= center < end),
                     boundaries[-1][2])
        result.append(0.0 if phone in UNVOICED else value)
    return result


def main() -> None:
    rows = json.loads(SOURCE.read_text())
    row = rows[0]
    timestep = float(row["f0_timestep"])
    values = mask_f0(row["ph_seq"].split(), list(map(float, row["ph_dur"].split())),
                     list(map(float, row["f0_seq"].split())), timestep)
    row["f0_seq"] = " ".join(f"{value:.3f}" for value in values)
    row.pop("velocity", None)
    row.pop("velocity_timestep", None)
    TARGET.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    report = {
        "status": "canonical_voicing_applied",
        "input": str(SOURCE.relative_to(ROOT)),
        "output": str(TARGET.relative_to(ROOT)),
        "frames": len(values),
        "voiced_ratio_before": round(sum(value > 0 for value in map(float, json.loads(SOURCE.read_text())[0]["f0_seq"].split())) / len(values), 4),
        "voiced_ratio_after": round(sum(value > 0 for value in values) / len(values), 4),
        "unvoiced_f0_zero": True,
        "production_controls": {"gender": 0, "velocity": 1},
    }
    output = ROOT / "artifacts/reports/diffsinger_rapid_voicing.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
