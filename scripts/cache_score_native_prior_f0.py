#!/usr/bin/env python3
"""Cache RMVPE F0 on the DiffSinger 44.1 kHz / 512-hop grid."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "data/cache/soulx-singer"
sys.path.insert(0, str(SOULX))
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def main() -> None:
    source = ROOT / "data/external/manifests/score_native_vocalset_realized.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    cache = ROOT / "data/cache/score_native_f0/vocalset"
    cache.mkdir(parents=True, exist_ok=True)
    extractor = F0Extractor(
        str(SOULX / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=44_100, hop_size=512, max_duration=300, verbose=False,
    )
    voiced = []
    for index, row in enumerate(rows, 1):
        target = cache / f"{row['id']}.npy"
        if not target.is_file():
            extractor.process(str(ROOT / row["audio_path"]), f0_path=str(target), verbose=False)
        f0 = np.load(target)
        voiced.append(float(np.mean(f0 > 0)))
        row["f0_path"] = str(target.relative_to(ROOT))
        row["f0_label_status"] = "RMVPE_inferred_acoustic_condition"
        row["f0_timestep_seconds"] = 512 / 44_100
        if index % 100 == 0:
            print(f"{index}/{len(rows)}", flush=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {
        "status": "pass", "rows": len(rows), "extractor": "SoulX RMVPE",
        "sample_rate": 44_100, "hop_size": 512,
        "mean_voiced_ratio": round(float(np.mean(voiced)), 4),
        "labels_are_inferred": True,
    }
    (ROOT / "artifacts/reports/score_native_prior_f0.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
