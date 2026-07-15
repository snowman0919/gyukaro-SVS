#!/usr/bin/env python3
"""Materialize the bounded VocalSet prior into ignored workspace storage."""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/work/VocalSet1-2.fixed.zip"
DEST = ROOT / "data/external/work/score_native_vocalset"


def extract(row: dict) -> dict:
    target = DEST / row["speaker"] / row["technique"].replace("/", "_") / (row["id"] + ".wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != row["uncompressed_bytes"]:
        with target.open("wb") as handle:
            result = subprocess.run(
                ("unzip", "-p", str(ARCHIVE), row["source_member"]),
                stdout=handle,
                stderr=subprocess.PIPE,
                env=os.environ | {"UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"},
            )
        # Info-ZIP returns 4 after successfully streaming members from the
        # repaired >4 GiB archive. Trust exact byte size plus libsndfile below.
        if target.stat().st_size != row["uncompressed_bytes"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(result.stderr.decode(errors="replace")[-500:])
    info = sf.info(target)
    if info.frames <= 0:
        raise RuntimeError(f"invalid audio: {target}")
    return row | {"audio_path": str(target.relative_to(ROOT)), "duration_seconds": round(info.duration, 6)}


def main() -> None:
    source = ROOT / "data/external/manifests/score_native_vocalset_prior.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    with ThreadPoolExecutor(max_workers=4) as pool:
        realized = list(pool.map(extract, rows))
    target = ROOT / "data/external/manifests/score_native_vocalset_realized.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in realized))
    report = {
        "status": "pass",
        "rows": len(realized),
        "hours": round(sum(row["duration_seconds"] for row in realized) / 3600, 3),
        "speakers": len({row["speaker"] for row in realized}),
        "source_audio_modified": False,
        "raw_audio_bundled": False,
    }
    (ROOT / "artifacts/reports/score_native_prior_extraction.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
