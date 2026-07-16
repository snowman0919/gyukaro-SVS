#!/usr/bin/env python3
"""Prepare a licensed rapid-coarticulation curriculum for the Japanese singer."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from analyze_diffsinger_rapid_coverage import (
    TARGET_REPEAT_SECONDS,
    mora_windows,
    normalized_phones,
    selected_rows,
)


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE_RAW = WORK / "raw/gtsinger_ja_soprano"
TARGET_RAW = WORK / "raw/gtsinger_ja_rapid_curriculum"
CHECKPOINT = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_source/model_ckpt_steps_15000.ckpt"


def rank_rows(rows: list[dict]) -> list[tuple[float, int, dict]]:
    ranked = []
    for index, row in enumerate(rows):
        windows = mora_windows(row)
        if windows:
            ranked.append((min(windows), index, row))
    return sorted(ranked, key=lambda value: (value[0], value[1]))


def curriculum(rows: list[dict]) -> dict:
    ranked = rank_rows(rows)
    exact = [value for value in ranked if value[0] <= TARGET_REPEAT_SECONDS]
    near = [value for value in ranked if value[0] > TARGET_REPEAT_SECONDS]
    validation = near[:8]
    near_train = near[8:48]
    used = {index for _, index, _ in exact + validation + near_train}
    remaining = [(index, row) for index, row in enumerate(rows) if index not in used]
    required = {phone for row in rows for phone in normalized_phones(row)}
    covered = {
        phone for _, _, row in exact + validation + near_train for phone in normalized_phones(row)
    }
    anchors: list[tuple[int, dict]] = []
    while required - covered and len(anchors) < 64:
        candidate = max(
            remaining,
            key=lambda value: len(set(normalized_phones(value[1])) & (required - covered)),
        )
        anchors.append(candidate)
        remaining.remove(candidate)
        covered.update(normalized_phones(candidate[1]))
    slots = 64 - len(anchors)
    if slots:
        stride = max(1, len(remaining) // slots)
        anchors.extend(remaining[::stride][:slots])
    return {
        "exact": exact,
        "near": near_train,
        "validation": validation,
        "anchors": [(None, index, row) for index, row in anchors],
    }


def write_raw(groups: dict) -> tuple[list[str], int]:
    source_rows = {
        row["name"]: row
        for row in csv.DictReader((SOURCE_RAW / "transcriptions.csv").open())
    }
    wavs = TARGET_RAW / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    validation_names: list[str] = []
    written = 0
    with (TARGET_RAW / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "ph_seq", "ph_dur"))
        writer.writeheader()
        for group, copies in (("exact", 8), ("near", 2), ("anchors", 1), ("validation", 1)):
            for _, source_index, _ in groups[group]:
                source_name = f"gtsja{source_index:04d}"
                source = source_rows[source_name]
                for copy in range(copies):
                    name = f"{group}_{source_name}_{copy:02d}"
                    target = wavs / f"{name}.wav"
                    if not target.exists():
                        target.symlink_to((SOURCE_RAW / "wavs" / f"{source_name}.wav").resolve())
                    writer.writerow({"name": name, "ph_seq": source["ph_seq"],
                                     "ph_dur": source["ph_dur"]})
                    written += 1
                    if group == "validation":
                        validation_names.append(name)
    return validation_names, written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "data/external/raw/gtsinger-lfs/processed/Japanese/metadata.json",
    )
    args = parser.parse_args()
    rows = selected_rows(json.loads(args.metadata.read_text()))
    groups = curriculum(rows)
    validation, entries = write_raw(groups)

    config = yaml.safe_load((ROOT / "configs/diffsinger_gtsinger_ja.yaml").read_text())
    config.update({
        "datasets": [{
            "raw_data_dir": str(TARGET_RAW), "speaker": "gts_ja_soprano", "spk_id": 0,
            "language": "gyu", "test_prefixes": validation,
        }],
        "binary_data_dir": str(WORK / "binary_gtsinger_ja_rapid_curriculum"),
        "finetune_enabled": True,
        "finetune_ckpt_path": str(CHECKPOINT),
        "finetune_strict_shapes": True,
        "frozen_params": ["model.diffusion"],
        "max_updates": 1200,
        "val_check_interval": 200,
        "num_ckpt_keep": 6,
        "optimizer_args": {"lr": 2e-5},
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True, "range": [-2.0, 5.0], "scale": .5},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": .5},
            "random_time_stretching": {"enabled": False, "range": [.95, 1.05], "scale": 1.0},
        },
    })
    config_path = ROOT / "configs/diffsinger_gtsinger_ja_rapid_curriculum.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "rapid_curriculum_ready_for_binarization",
        "source": "GTSinger Japanese soprano manual phoneme alignments (CC BY-NC-SA 4.0)",
        "target_phrase_audio_used_for_training": False,
        "unique_exact_speed_rows": len(groups["exact"]),
        "unique_near_speed_rows": len(groups["near"]),
        "unique_anchor_rows": len(groups["anchors"]),
        "unique_validation_rows": len(groups["validation"]),
        "training_entries_after_explicit_oversampling": entries - len(validation),
        "validation_entries": len(validation),
        "oversampling": {"exact_speed": 8, "near_speed": 2, "anchors": 1},
        "foundation_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "trainable_path": "phoneme/F0 encoder and auxiliary acoustic decoder",
        "frozen_path": "diffusion decoder",
        "learning_rate": 2e-5,
        "max_updates": 1200,
        "config": str(config_path.relative_to(ROOT)),
        "decision_rule": "waveform, free Japanese Whisper, pitch and human listening gate",
        "release_allowed": False,
    }
    output = ROOT / "artifacts/reports/diffsinger_gtsinger_ja_rapid_curriculum.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
