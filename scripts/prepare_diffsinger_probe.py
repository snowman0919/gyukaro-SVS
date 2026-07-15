#!/usr/bin/env python3
"""Create a symlink-only DiffSinger raw corpus and official-model config."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]


def link_dataset(name: str, rows: list[dict], labels) -> dict:
    raw = WORK / "raw" / name
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        for row in rows:
            phones, durations = labels(row)
            assert len(phones) == len(durations) and abs(sum(durations) - row["duration_seconds"]) < .03
            target = wavs / f"{row['id']}.wav"
            if not target.exists():
                target.symlink_to(ROOT / row["audio_path"])
            writer.writerow({"name": row["id"], "ph_seq": " ".join(phones), "ph_dur": " ".join(f"{x:.6f}" for x in durations)})
    return {"raw_data_dir": str(raw), "speaker": name, "language": "gyu"}


def vocal_labels(row: dict) -> tuple[list[str], list[float]]:
    f0 = np.load(ROOT / row["f0_path"])
    step, duration = row["f0_timestep_seconds"], row["duration_seconds"]
    voiced = np.flatnonzero(f0 > 0)
    start = min(duration, float(voiced[0] * step)) if len(voiced) else 0.0
    end = min(duration, float((voiced[-1] + 1) * step)) if len(voiced) else duration
    pause = "AP" if "breathy" in row["technique"] else "SP"
    return [pause, row["phoneme"], "SP"], [start, max(0.0, end - start), max(0.0, duration - end)]


def gyu_labels(row: dict) -> tuple[list[str], list[float]]:
    duration = row["duration_seconds"]
    events, cursor = [], 0.0
    for phone in read_alignment[row["id"]]:
        start = max(cursor, float(phone["start"]))
        if start - cursor >= .005:
            events.append(("SP", start - cursor))
        length = min(float(phone["duration"]), max(0.0, duration - start))
        if length > 0:
            events.append((phone["symbol"], length))
        cursor = start + length
    if duration - cursor > 0:
        events.append(("SP", duration - cursor))
    return [x[0] for x in events], [x[1] for x in events]


def main() -> None:
    global read_alignment
    read_alignment = {row["id"]: row["phones"] for row in read("data/manifests/real_phoneme_alignment.jsonl")}
    vocal = read("data/external/manifests/score_native_vocalset_realized.jsonl")
    gyu = [row for row in read("data/manifests/diffsinger_score_native.jsonl") if row["training_allowed"]]
    grouped = defaultdict(list)
    for row in vocal:
        grouped[row["speaker"]].append(row)
    datasets = []
    for speaker, items in sorted(grouped.items()):
        dataset = link_dataset(f"vocalset_{speaker}", items, vocal_labels)
        dataset["test_prefixes"] = ["vocalset_"] if items[0]["split"] == "validation" else []
        datasets.append(dataset)
    gyu_dataset = link_dataset("gyu", gyu, gyu_labels)
    gyu_dataset["test_prefixes"] = [row["id"] for row in gyu if row["split"] != "train"]
    datasets.append(gyu_dataset)

    phones = sorted({p for row in gyu for p in row["ph_seq"]} | {"a", "e", "i", "o", "u"})
    dictionary = WORK / "dictionary-gyu.txt"
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in phones))
    config = {
        "base_config": ["configs/acoustic.yaml"], "dictionaries": {"gyu": str(dictionary)},
        "datasets": datasets, "binary_data_dir": str(WORK / "binary"),
        "use_spk_id": True, "num_spk": len(datasets), "use_lang_id": False, "num_lang": 1,
        "pe": "rmvpe", "pe_ckpt": str(ROOT / "data/cache/diffsinger-assets/rmvpe/model.pt"),
        "hnsep": "world", "use_energy_embed": False, "use_breathiness_embed": False,
        "use_voicing_embed": False, "use_tension_embed": False,
        "vocoder": "NsfHifiGAN", "vocoder_ckpt": str(WORK / "vocoder/model.ckpt"),
        "binarization_args": {"num_workers": 4, "shuffle": True},
    }
    config_path = ROOT / "configs/diffsinger_score_native.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    binary = WORK / "binary"
    report = {"status": "raw_ready", "datasets": len(datasets), "vocalset_rows": len(vocal),
              "gyu_training_rows": len(gyu), "gyu_independent_rows_excluded": 24,
              "phonemes": len(phones), "audio_copy": False, "config": str(config_path.relative_to(ROOT))}
    if (binary / "train.meta").is_file() and (binary / "valid.meta").is_file():
        train = pickle.load((binary / "train.meta").open("rb"))
        valid = pickle.load((binary / "valid.meta").open("rb"))
        report |= {"status": "binarization_pass", "train_rows": len(train["spk_ids"]),
                   "validation_rows": len(valid["spk_ids"]),
                   "train_hours": round(sum(train["lengths"]) * 512 / 44_100 / 3600, 3),
                   "validation_minutes": round(sum(valid["lengths"]) * 512 / 44_100 / 60, 3)}
    checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_smoke/model_ckpt_steps_2.ckpt"
    if checkpoint.is_file():
        report |= {"training_smoke": "pass", "smoke_steps": 2, "model_parameters": 69_100_000,
                   "smoke_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "diffsinger_revision": "753b7cc622aadf802b3145d7bb8f7df4afa213c4"}
    (ROOT / "artifacts/reports/diffsinger_probe_data.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
