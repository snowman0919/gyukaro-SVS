#!/usr/bin/env python3
"""Create a no-duplication GYU-only DiffSinger adaptation dataset."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/work/diffsinger_score_native/binary"
TARGET = ROOT / "data/external/work/diffsinger_score_native/binary_gyu"
GYU_SPEAKER_ID = 20


def subset(split: str) -> int:
    metadata = pickle.loads((SOURCE / f"{split}.meta").read_bytes())
    selected = [i for i, speaker in enumerate(metadata["spk_ids"]) if speaker == GYU_SPEAKER_ID]
    with h5py.File(SOURCE / f"{split}.data", "r") as source, h5py.File(TARGET / f"{split}.data", "w") as target:
        for output_index, source_index in enumerate(selected):
            source.copy(str(source_index), target, name=str(output_index))
    filtered = {key: [values[i] for i in selected] for key, values in metadata.items()}
    (TARGET / f"{split}.meta").write_bytes(pickle.dumps(filtered))
    return len(selected)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    counts = {split: subset(split) for split in ("train", "valid")}
    for name in ("dictionary-gyu.txt", "lang_map.json", "spk_map.json"):
        shutil.copy2(SOURCE / name, TARGET / name)
    report = {
        "status": "ready",
        "speaker": "gyu",
        "speaker_id": GYU_SPEAKER_ID,
        "rows": counts,
        "duplicates_added": 0,
        "source": str(SOURCE.relative_to(ROOT)),
        "target": str(TARGET.relative_to(ROOT)),
    }
    (TARGET / "subset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
