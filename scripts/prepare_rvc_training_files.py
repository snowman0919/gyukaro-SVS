#!/usr/bin/env python3
"""Create the RVC v2 training config and deterministic single-speaker file list."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RVC = ROOT / "data/cache/rvc"


def main() -> None:
    experiment = RVC / "logs/gyu_score_native_source"
    folders = {
        "wav": experiment / "0_gt_wavs",
        "feature": experiment / "3_feature768",
        "f0": experiment / "2a_f0",
        "f0nsf": experiment / "2b-f0nsf",
    }
    names = set.intersection(*(
        {path.name.split(".")[0] for path in folder.iterdir()}
        for folder in folders.values()
    ))
    rows = [
        "|".join([
            str(folders["wav"] / f"{name}.wav"),
            str(folders["feature"] / f"{name}.npy"),
            str(folders["f0"] / f"{name}.wav.npy"),
            str(folders["f0nsf"] / f"{name}.wav.npy"),
            "0",
        ])
        for name in sorted(names)
    ]
    (experiment / "filelist.txt").write_text("\n".join(rows) + "\n")
    shutil.copyfile(RVC / "configs/v2/48k.json", experiment / "config.json")
    report = {
        "status": "rvc_training_files_ready",
        "segments": len(rows),
        "version": "v2",
        "sample_rate": "48k",
        "f0": True,
        "feature": "facebook/hubert-base-ls960 Transformers last hidden, 768d",
        "feature_equivalence": "architecture-compatible probe; not bit-identical fairseq extraction",
    }
    (ROOT / "artifacts/reports/rvc_gyu_training_files.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
