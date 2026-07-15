#!/usr/bin/env python3
"""Derive contiguous GYU singing segments without modifying source recordings."""
from __future__ import annotations

import csv
import argparse
import copy
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
GAP_SPLIT_SECONDS = 0.25
LEADING_CONTEXT_SECONDS = 0.02
TRAILING_CONTEXT_SECONDS = 0.03


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dictionary_tokens(path: Path) -> list[str]:
    values = {"AP", "SP"}
    values.update(line.split("\t", 1)[0] for line in path.read_text().splitlines() if line)
    return sorted(values)


def remap_vocabulary(source: Path, old_dictionary: Path, new_dictionary: Path, target: Path) -> dict:
    old_tokens = dictionary_tokens(old_dictionary)
    new_tokens = dictionary_tokens(new_dictionary)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    key = "model.fs2.txt_embed.weight"
    old_embedding = state[key]
    old_ids = {token: index + 1 for index, token in enumerate(old_tokens)}
    new_ids = {token: index + 1 for index, token in enumerate(new_tokens)}
    new_embedding = old_embedding.new_empty((len(new_tokens) + 1, old_embedding.shape[1]))
    new_embedding[0] = old_embedding[0]
    initialized = {}
    for token, index in new_ids.items():
        if token in old_ids:
            new_embedding[index] = old_embedding[old_ids[token]]
            continue
        category = token.rsplit("_", 1)[0]
        sources = [value for value in old_tokens if value.startswith(category + "_")]
        if not sources:
            raise RuntimeError(f"no same-category embedding source for {token}")
        new_embedding[index] = old_embedding[[old_ids[value] for value in sources]].mean(dim=0)
        initialized[token] = f"mean:{category}"
    state[key] = new_embedding
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "category": checkpoint.get("category")}, target)
    shared_error = max(
        float((new_embedding[new_ids[token]] - old_embedding[old_ids[token]]).abs().max())
        for token in old_ids.keys() & new_ids.keys()
    )
    return {
        "old_vocabulary": len(old_tokens) + 1,
        "new_vocabulary": len(new_tokens) + 1,
        "new_token_initialization": initialized,
        "shared_embedding_max_abs_error": shared_error,
        "checkpoint": str(target.relative_to(ROOT)),
        "checkpoint_sha256": sha256(target),
    }


def split_phones(phones: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    cursor = 0.0
    for phone in phones:
        start = float(phone["start"])
        if current and start - cursor >= GAP_SPLIT_SECONDS:
            groups.append(current)
            current = []
        current.append(phone)
        cursor = start + float(phone["duration"])
    if current:
        groups.append(current)
    return groups


def segment_labels(group: list[dict], start: float, end: float) -> tuple[list[str], list[float]]:
    events: list[tuple[str, float]] = []
    cursor = start
    for phone in group:
        phone_start = max(cursor, float(phone["start"]))
        if phone_start - cursor >= 0.005:
            events.append(("SP", phone_start - cursor))
        phone_end = min(end, phone_start + float(phone["duration"]))
        if phone_end > phone_start:
            events.append((phone["symbol"], phone_end - phone_start))
        cursor = phone_end
    if end - cursor >= 0.005:
        events.append(("SP", end - cursor))
    drift = end - start - sum(duration for _, duration in events)
    events[-1] = (events[-1][0], events[-1][1] + drift)
    return [symbol for symbol, _ in events], [duration for _, duration in events]


def corpus(all_recordings: bool) -> tuple[dict[str, dict], dict[str, list[dict]], str]:
    score_rows = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data/manifests/diffsinger_score_native.jsonl")
    }
    if not all_recordings:
        rows = {identifier: row for identifier, row in score_rows.items() if row["training_allowed"]}
        path = ROOT / "data/manifests/real_phoneme_alignment.jsonl"
        name = "gyu_segmented"
    else:
        recordings = {
            row["id"]: row
            for row in read_jsonl(ROOT / "data/manifests/real_recordings.jsonl")
        }
        allowed = {
            row["id"]
            for row in read_jsonl(ROOT / "data/manifests/real_phoneme_alignment_all.jsonl")
        }
        rows = {}
        for identifier in sorted(allowed):
            recording = recordings[identifier]
            score = score_rows.get(identifier, {})
            rows[identifier] = {
                "id": identifier,
                "audio_path": recording["pcm_master"],
                "duration_seconds": recording["duration_sec"],
                "split": score.get("split", "validation" if recording["source_index"] % 5 == 0 else "train"),
                "script_block": recording["script_block"],
            }
        path = ROOT / "data/manifests/real_phoneme_alignment_all.jsonl"
        name = "gyu_all_segmented"
    alignments = {
        row["id"]: row["phones"]
        for row in read_jsonl(path)
    }
    return rows, alignments, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-recordings", action="store_true")
    args = parser.parse_args()
    rows, alignments, name = corpus(args.all_recordings)
    raw = WORK / "raw" / name
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    derived: list[dict] = []
    transcription_rows: list[dict] = []
    for source_id, row in rows.items():
        source = ROOT / row["audio_path"]
        source_hash = sha256(source)
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        duration = len(audio) / sample_rate
        for index, group in enumerate(split_phones(alignments[source_id])):
            start = max(0.0, float(group[0]["start"]) - LEADING_CONTEXT_SECONDS)
            end = min(duration, float(group[-1]["start"]) + float(group[-1]["duration"]) + TRAILING_CONTEXT_SECONDS)
            phones, phone_durations = segment_labels(group, start, end)
            identifier = f"{source_id}_seg{index:02d}"
            target = wavs / f"{identifier}.wav"
            first = round(start * sample_rate)
            last = round(end * sample_rate)
            sf.write(target, audio[first:last], sample_rate, subtype="PCM_24")
            actual_duration = (last - first) / sample_rate
            phone_durations[-1] += actual_duration - sum(phone_durations)
            assert abs(sum(phone_durations) - actual_duration) < 1e-6
            transcription_rows.append({
                "name": identifier,
                "ph_seq": " ".join(phones),
                "ph_dur": " ".join(f"{value:.6f}" for value in phone_durations),
            })
            derived.append({
                "id": identifier,
                "source_id": source_id,
                "audio_path": str(target.relative_to(ROOT)),
                "source_audio_sha256": source_hash,
                "sample_rate": sample_rate,
                "duration_seconds": actual_duration,
                "source_start_seconds": start,
                "source_end_seconds": end,
                "ph_seq": phones,
                "ph_dur": phone_durations,
                "split": row["split"],
                "script_block": row.get("script_block"),
                "label_status": "inferred_from_singing_aware_alignment",
                "training_role": "real_gyu_singing_contiguous_acoustic_segment",
                "source_recording_modified": False,
            })

    if args.all_recordings:
        for derived_row, transcription in zip(derived, transcription_rows):
            if "SP" not in derived_row["ph_seq"]:
                continue
            index = derived_row["ph_seq"].index("SP")
            derived_row["ph_seq"][index] = "AP"
            transcription["ph_seq"] = " ".join(derived_row["ph_seq"])
            break

    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(transcription_rows)
    manifest = ROOT / f"data/manifests/diffsinger_{name}.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in derived))

    base = yaml.safe_load((ROOT / "configs/diffsinger_score_native.yaml").read_text())
    source_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000.ckpt"
    vocabulary = None
    if args.all_recordings:
        old_dictionary = Path(base["dictionaries"]["gyu"])
        dictionary = WORK / "dictionary-gyu-all-segmented.txt"
        phones = {phone for row in derived for phone in row["ph_seq"]} | {"a", "e", "i", "o", "u"}
        dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP"})))
        target_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000_gyu_all_vocab.ckpt"
        vocabulary = remap_vocabulary(source_checkpoint, old_dictionary, dictionary, target_checkpoint)
        base["dictionaries"]["gyu"] = str(dictionary)
    else:
        target_checkpoint = source_checkpoint
    gyu = {
        "raw_data_dir": str(raw),
        "speaker": "gyu",
        "language": "gyu",
        "spk_id": 20,
        "test_prefixes": [row["id"] for row in derived if row["split"] != "train"],
    }
    base["datasets"] = [*base["datasets"][:-1], gyu]
    base["binary_data_dir"] = str(WORK / f"binary_{name}")
    prior_config = ROOT / f"configs/diffsinger_{name}_prior.yaml"
    prior_config.write_text(yaml.safe_dump(base, sort_keys=False))

    finetune = copy.deepcopy(base)
    finetune.update({
        "finetune_enabled": True,
        "finetune_ckpt_path": str(target_checkpoint),
        "finetune_ignored_params": [],
        "finetune_strict_shapes": True,
        "freezing_enabled": True,
        "frozen_params": [
            "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.fs2.dur_embed", "model.fs2.pitch_embed",
        ],
        "max_updates": 600,
        "val_check_interval": 200,
        "num_valid_plots": 0,
        "val_with_vocoder": False,
        "max_batch_frames": 20_000,
        "max_batch_size": 8,
        "num_ckpt_keep": 3,
    })
    finetune.setdefault("optimizer_args", {})["lr"] = 0.00002
    finetune_config = ROOT / f"configs/diffsinger_{name}_finetune.yaml"
    finetune_config.write_text(yaml.safe_dump(finetune, sort_keys=False))
    adaptation_config = None
    if args.all_recordings:
        adaptation = copy.deepcopy(finetune)
        adaptation["datasets"] = [gyu]
        adaptation["binary_data_dir"] = str(WORK / "binary_gyu_all_adapt")
        adaptation_dictionary = WORK / "dictionary-gyu-all-adapt.txt"
        adaptation_phones = {phone for row in derived for phone in row["ph_seq"]}
        adaptation_dictionary.write_text("".join(
            f"{phone}\t{phone}\n" for phone in sorted(adaptation_phones - {"AP", "SP"})
        ))
        adaptation_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000_gyu_adapt_vocab.ckpt"
        remap_vocabulary(target_checkpoint, dictionary, adaptation_dictionary, adaptation_checkpoint)
        adaptation["dictionaries"]["gyu"] = str(adaptation_dictionary)
        adaptation["finetune_ckpt_path"] = str(adaptation_checkpoint)
        adaptation["max_updates"] = 300
        adaptation["val_check_interval"] = 100
        adaptation["optimizer_args"]["lr"] = 0.00001
        adaptation_config = ROOT / "configs/diffsinger_gyu_all_adapt.yaml"
        adaptation_config.write_text(yaml.safe_dump(adaptation, sort_keys=False))

    total = sum(row["duration_seconds"] for row in derived)
    silence = sum(
        duration
        for row in derived
        for phone, duration in zip(row["ph_seq"], row["ph_dur"])
        if phone == "SP"
    )
    report = {
        "status": "contiguous_real_gyu_segments_ready",
        "all_usable_recordings": args.all_recordings,
        "source_rows": len(rows),
        "segments": len(derived),
        "train_segments": sum(row["split"] == "train" for row in derived),
        "validation_segments": sum(row["split"] != "train" for row in derived),
        "duration_minutes": round(total / 60, 3),
        "silence_fraction": round(silence / total, 6),
        "previous_full_phrase_silence_fraction": 0.6238,
        "gap_split_seconds": GAP_SPLIT_SECONDS,
        "labels": "inferred; independent-score evaluation rows remain excluded",
        "source_recordings_modified": False,
        "manifest": str(manifest.relative_to(ROOT)),
        "prior_config": str(prior_config.relative_to(ROOT)),
        "finetune_config": str(finetune_config.relative_to(ROOT)),
        "adaptation_config": str(adaptation_config.relative_to(ROOT)) if adaptation_config else None,
        "vocabulary_remap": vocabulary,
    }
    output = ROOT / f"artifacts/reports/diffsinger_{name}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
