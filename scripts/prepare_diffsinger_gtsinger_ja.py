#!/usr/bin/env python3
"""Prepare the CC BY-NC-SA GTSinger Japanese soprano DiffSinger foundation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess

import h5py
import numpy as np
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data/external"
DATASET = EXTERNAL / "raw/gtsinger-lfs"
WORK = EXTERNAL / "work/diffsinger_score_native"
REPO = "GTSinger/GTSinger"
REVISION = "4426c862beed558b7e1cb8a4dce7e8c0c83bb208"
CODE_REVISION = "0619d61d5301c4340db442a15cf3e73e197e9101"
SINGER = "JA-Soprano-1"


def normalized_phones(row: dict) -> list[str]:
    return ["SP" if phone == "<SP>" else "AP" if phone == "<AP>" else phone
            for phone in row["ph"]]


def selected_rows(rows: list[dict], singer: str = SINGER) -> list[dict]:
    return [row for row in rows if row["language"] == "Japanese" and row["singer"] == singer
            and "#Control_Group#" in row["item_name"]]


def heldout_names(rows: list[dict], song_count: int = 2) -> list[str]:
    """Hold out complete songs so styled duplicates cannot cross the split."""
    songs = sorted({row["item_name"].split("#")[3] for row in rows})[-song_count:]
    return [f"gtsja{index:04d}" for index, row in enumerate(rows)
            if row["item_name"].split("#")[3] in songs]


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def download() -> None:
    """Use Git LFS batching; per-file Hub HEAD requests are rate-limited."""
    if not (DATASET / ".git").is_dir():
        environment = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}
        run("git", "clone", "--depth", "1", "--no-checkout",
            f"https://huggingface.co/datasets/{REPO}", str(DATASET), env=environment)
        run("git", "fetch", "--depth", "1", "origin", REVISION, cwd=DATASET)
        run("git", "sparse-checkout", "init", "--no-cone", cwd=DATASET)
        run("git", "sparse-checkout", "set", "Japanese/JA-Soprano-1/**/Control_Group/*.wav",
            "processed/Japanese/metadata.json", "dataset_license.md", cwd=DATASET)
        run("git", "checkout", REVISION, cwd=DATASET, env=environment)
    run("git", "lfs", "pull", "--include=Japanese/JA-Soprano-1/**/Control_Group/*.wav",
        cwd=DATASET)
    run("git", "lfs", "pull", "--include=processed/Japanese/metadata.json", cwd=DATASET)


def ensure_metadata(download_audio: bool) -> Path:
    if download_audio:
        download()
    DATASET.mkdir(parents=True, exist_ok=True)
    for filename in ("processed/Japanese/metadata.json", "dataset_license.md"):
        target = DATASET / filename
        if not target.is_file():
            cached = hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(cached)
    metadata = DATASET / "processed/Japanese/metadata.json"
    return metadata


def write_training_data(rows: list[dict]) -> tuple[Path, set[str], float]:
    raw = WORK / "raw/gtsinger_ja_soprano"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    phones: set[str] = set()
    seconds = 0.0
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "ph_seq", "ph_dur"))
        writer.writeheader()
        for index, row in enumerate(rows):
            name = f"gtsja{index:04d}"
            source = DATASET / row["wav_fn"]
            if not source.is_file():
                raise FileNotFoundError(f"run with --download first: {source}")
            sequence = normalized_phones(row)
            durations = [float(value) for value in row["ph_durs"]]
            audio_duration = sf.info(source).duration
            delta = audio_duration - sum(durations)
            if delta > .001:
                sequence.append("SP")
                durations.append(delta)
            elif delta < -.001:
                durations[-1] += delta
            if durations[-1] <= 0 or abs(sum(durations) - audio_duration) >= .002:
                raise ValueError(f"invalid alignment for {row['item_name']}: {delta:+.4f}s")
            target = wavs / f"{name}.wav"
            if not target.exists():
                target.symlink_to(source)
            writer.writerow({
                "name": name,
                "ph_seq": " ".join(sequence),
                "ph_dur": " ".join(f"{value:.7f}" for value in durations),
            })
            phones.update(sequence)
            seconds += audio_duration
    return raw, phones, seconds


def write_config(raw: Path, phones: set[str], rows: list[dict], seconds: float) -> dict:
    dictionary = WORK / "dictionary-gtsinger-ja.txt"
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP"})))
    base = yaml.safe_load((ROOT / "configs/diffsinger_pjs_compact.yaml").read_text())
    validation = heldout_names(rows)
    base.update({
        "dictionaries": {"gyu": str(dictionary)},
        "datasets": [{
            "raw_data_dir": str(raw), "speaker": "gts_ja_soprano", "spk_id": 0,
            "language": "gyu", "test_prefixes": validation,
        }],
        "binary_data_dir": str(WORK / "binary_gtsinger_ja_soprano"),
        "binarizer_cls": "diffsinger_neutral_augmentation_binarizer.NeutralAugmentationBinarizer",
        "finetune_enabled": False, "finetune_ckpt_path": None,
        "finetune_strict_shapes": True, "frozen_params": ["model.diffusion"],
        "hidden_size": 256, "num_heads": 4, "enc_layers": 6,
        "backbone_args": {
            "num_channels": 512, "num_layers": 6, "kernel_size": 15,
            "dropout_rate": 0.0, "use_conditioner_cache": True, "glu_type": "atanglu",
        },
        "shallow_diffusion_args": {
            "train_aux_decoder": True, "train_diffusion": False,
            "aux_decoder_arch": "convnext", "aux_decoder_grad": .1,
            "aux_decoder_args": {
                "num_channels": 384, "num_layers": 6, "kernel_size": 7, "dropout_rate": .1,
            },
        },
        "use_key_shift_embed": True, "use_speed_embed": True,
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True, "range": [-2.0, 12.0], "scale": 1.0},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": .5},
            "random_time_stretching": {"enabled": True, "range": [.95, 4.5], "scale": 2.0},
        },
        "max_updates": 15_000, "val_check_interval": 1_000, "num_ckpt_keep": 5,
        "max_batch_frames": 16_000, "max_batch_size": 6,
        "optimizer_args": {"lr": 3e-4},
    })
    config = ROOT / "configs/diffsinger_gtsinger_ja.yaml"
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    fast = [row for row in rows if row["pace"] == "fast"]
    report = {
        "status": "ready_for_binarization",
        "dataset": "GTSinger Japanese soprano neutral Control_Group", "license": "CC BY-NC-SA 4.0",
        "redistribution": "derived checkpoint/package must remain non-commercial share-alike",
        "rows": len(rows), "validation_rows": len(validation),
        "duration_hours": round(seconds / 3600, 4), "fast_rows": len(fast),
        "fast_duration_hours": round(sum(sum(row["ph_durs"]) for row in fast) / 3600, 4),
        "phones": len(phones), "speaker": SINGER,
        "dataset_revision": REVISION, "official_code_revision": CODE_REVISION,
        "dataset_license_sha256": hashlib.sha256((DATASET / "dataset_license.md").read_bytes()).hexdigest(),
        "alignment": "dataset-provided manual phoneme alignment",
        "score": "dataset-provided realistic MusicXML; acoustic training uses aligned phones and extracted real F0",
        "control_policy": "pitch/rate augmented examples keep neutral speaker controls",
        "architecture": "medium 256-dim 6-layer phoneme encoder and 384-channel auxiliary decoder",
        "config": str(config.relative_to(ROOT)),
        "decision_rule": "exact rapid lexical/F0/voicing gate before GYU adaptation",
    }
    binary = Path(base["binary_data_dir"])
    if (binary / "train.data").is_file() and (binary / "valid.meta").is_file():
        with h5py.File(binary / "train.data") as dataset:
            zero_ratios = [float(np.mean(dataset[key]["f0"][()] == 0)) for key in dataset]
            controls = {
                name: sorted({float(dataset[key][name][()]) for key in dataset})
                for name in ("key_shift", "speed")
            }
        valid = pickle.load((binary / "valid.meta").open("rb"))
        report.update({
            "status": "binarization_pass", "binarized_train_rows": len(zero_ratios),
            "binarized_validation_rows": len(valid["lengths"]),
            "mean_zero_f0_ratio": round(float(np.mean(zero_ratios)), 4),
            "binarized_controls": controls,
        })
    checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_source/model_ckpt_steps_15000.ckpt"
    evaluation = ROOT / "artifacts/reports/diffsinger_gtsinger_ja_source_evaluation_11k_15k.json"
    if checkpoint.is_file():
        report.update({
            "source_checkpoint": str(checkpoint.relative_to(ROOT)),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_checkpoint_step": 15_000,
        })
    if evaluation.is_file():
        result = json.loads(evaluation.read_text())
        report["objective_rapid_gate"] = result["status"]
        report["human_listening"] = "pending"
    output = ROOT / "artifacts/reports/diffsinger_gtsinger_ja.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    metadata = ensure_metadata(args.download)
    rows = selected_rows(json.loads(metadata.read_text()))
    raw, phones, seconds = write_training_data(rows)
    print(json.dumps(write_config(raw, phones, rows, seconds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
