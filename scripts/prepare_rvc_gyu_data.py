#!/usr/bin/env python3
"""Prepare an isolated real-GYU timbre corpus for the bounded RVC probe."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/cache/rvc_gyu_input"


def main() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/manifests/real_recordings.jsonl").read_text().splitlines()
        if line
    ]
    selected = [
        row for row in rows
        if not row["corrupt"]
        and not row["clipping"]
        and row["active_voice_duration_sec"] >= 2.0
        and not 171 <= row["source_index"] <= 194
    ]
    INPUT.mkdir(parents=True, exist_ok=True)
    keep = set()
    for row in selected:
        target = INPUT / f'{row["id"]}.wav'
        source = (ROOT / row["pcm_master"]).resolve()
        keep.add(target.name)
        if target.is_symlink() and target.resolve() == source:
            continue
        target.unlink(missing_ok=True)
        target.symlink_to(source)
    for stale in INPUT.glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    report = {
        "status": "rvc_timbre_probe_data_ready",
        "rows": len(selected),
        "duration_minutes": round(sum(row["duration_sec"] for row in selected) / 60, 3),
        "active_voice_minutes": round(
            sum(row["active_voice_duration_sec"] for row in selected) / 60, 3
        ),
        "source": "original lossless-derived PCM masters; no denoise or dereverb",
        "independent_verified_score_rows_used": 0,
        "source_recordings_modified": False,
        "model": "RVC v2 48k f0",
        "model_revision": "7ef19867780cf703841ebafb565a4e47d1ea86ff",
        "model_license": "MIT",
    }
    (ROOT / "artifacts/reports/rvc_gyu_data.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
