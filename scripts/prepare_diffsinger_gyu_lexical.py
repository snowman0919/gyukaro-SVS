#!/usr/bin/env python3
"""Filter real-GYU segments to lexical singing phrases for bounded adaptation."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import yaml

from prepare_diffsinger_gyu_segments import remap_vocabulary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def is_lexical_phrase(source: dict) -> bool:
    hangul = re.sub(r"[^가-힣]", "", source["text"])
    diversity = len(set(hangul)) / max(1, len(hangul))
    return (
        source["segment_type"] == "phrase"
        and len(set(hangul)) >= 5
        and diversity >= 0.08
    )


def main() -> None:
    source_rows = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data/manifests/real_segments.jsonl")
    }
    all_segments = read_jsonl(ROOT / "data/manifests/diffsinger_gyu_all_segmented.jsonl")
    selected = [
        row for row in all_segments
        if is_lexical_phrase(source_rows[row["source_id"]])
    ]
    if not selected:
        raise RuntimeError("no lexical real-GYU segments selected")
    for row in selected:
        if "SP" in row["ph_seq"]:
            index = row["ph_seq"].index("SP")
            row["ph_seq"][index] = "AP"
            break

    raw = WORK / "raw/gyu_lexical_segmented"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    keep = {f'{row["id"]}.wav' for row in selected}
    for stale in wavs.glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    for row in selected:
        source = ROOT / row["audio_path"]
        target = wavs / source.name
        if target.exists():
            target.unlink()
        os.link(source, target)

    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "name": row["id"],
                "ph_seq": " ".join(row["ph_seq"]),
                "ph_dur": " ".join(f"{value:.6f}" for value in row["ph_dur"]),
            })

    manifest = ROOT / "data/manifests/diffsinger_gyu_lexical.jsonl"
    manifest.write_text("".join(
        json.dumps({
            **row,
            "selection": "source_segment_type_phrase_and_lexical_diversity_gate",
            "label_status": "inferred_from_singing_aware_alignment",
        }, ensure_ascii=False) + "\n"
        for row in selected
    ))

    config = yaml.safe_load((ROOT / "configs/diffsinger_gyu_all_adapt.yaml").read_text())
    old_dictionary = Path(config["dictionaries"]["gyu"])
    dictionary = WORK / "dictionary-gyu-lexical.txt"
    phones = {phone for row in selected for phone in row["ph_seq"]}
    dictionary.write_text("".join(
        f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP"})
    ))
    source_checkpoint = Path(config["finetune_ckpt_path"])
    target_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_gyu_lexical_vocab.ckpt"
    remap = remap_vocabulary(
        source_checkpoint, old_dictionary, dictionary, target_checkpoint
    )
    config["datasets"] = [{
        "raw_data_dir": str(raw),
        "speaker": "gyu",
        "language": "gyu",
        "spk_id": 20,
        "test_prefixes": [row["id"] for row in selected if row["split"] != "train"],
    }]
    config["binary_data_dir"] = str(WORK / "binary_gyu_lexical")
    config["dictionaries"]["gyu"] = str(dictionary)
    config["finetune_ckpt_path"] = str(target_checkpoint)
    config["max_updates"] = 300
    config["val_check_interval"] = 100
    config["optimizer_args"]["lr"] = 5e-6
    output_config = ROOT / "configs/diffsinger_gyu_lexical.yaml"
    output_config.write_text(yaml.safe_dump(config, sort_keys=False))

    selected_sources = {row["source_id"] for row in selected}
    excluded = [row for row in all_segments if row["id"] not in {item["id"] for item in selected}]
    report = {
        "status": "lexical_real_gyu_subset_ready",
        "source_segments": len(all_segments),
        "selected_segments": len(selected),
        "selected_sources": len(selected_sources),
        "train_segments": sum(row["split"] == "train" for row in selected),
        "validation_segments": sum(row["split"] != "train" for row in selected),
        "selected_minutes": round(sum(row["duration_seconds"] for row in selected) / 60, 3),
        "excluded_segments": len(excluded),
        "excluded_minutes": round(sum(row["duration_seconds"] for row in excluded) / 60, 3),
        "excluded_reason": "recording exercise or low-diversity repeated syllable source",
        "independent_score_rows_included": False,
        "labels": "inferred singing-aware CTC timing",
        "source_recordings_modified": False,
        "learning_rate": config["optimizer_args"]["lr"],
        "vocabulary_remap": remap,
        "manifest": str(manifest.relative_to(ROOT)),
        "config": str(output_config.relative_to(ROOT)),
    }
    (ROOT / "artifacts/reports/diffsinger_gyu_lexical.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
