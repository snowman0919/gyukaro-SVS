#!/usr/bin/env python3
"""Repair false CTC pauses in the bounded Fish/MOSS Korean speech prior."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf
import yaml

from prepare_diffsinger_gyu_coarticulated import split_phone_durations
from prepare_diffsinger_gyu_segments import remap_vocabulary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/manifests/diffsinger_ko_phoneme_prior.jsonl"
GYU = ROOT / "data/manifests/diffsinger_gyu_coarticulated.jsonl"
MAX_INTERSYLLABLE_BLANK_SECONDS = 0.45
LEADING_CONTEXT_SECONDS = 0.02
TRAILING_CONTEXT_SECONDS = 0.03


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def timed_phones(row: dict) -> list[dict]:
    cursor = 0.0
    phones = []
    for symbol, duration in zip(row["ph_seq"], row["ph_dur"]):
        duration = float(duration)
        if symbol not in {"SP", "AP"}:
            phones.append({"symbol": symbol, "start": cursor, "duration": duration})
        cursor += duration
    return phones


def syllable_groups(phones: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for phone in phones:
        if phone["symbol"].startswith("ko_onset_") and current:
            groups.append(current)
            current = []
        current.append(phone)
    if current:
        groups.append(current)
    if any(not any(phone["symbol"].startswith("ko_nucleus_") for phone in group) for group in groups):
        raise RuntimeError("Korean syllable group without nucleus")
    return groups


def labels(groups: list[list[dict]], crop_start: float, crop_end: float) -> tuple[list[str], list[float]]:
    events: list[tuple[str, float]] = []
    cursor = crop_start
    for index, group in enumerate(groups):
        group_start = max(cursor, float(group[0]["start"]))
        group_end = max(float(phone["start"]) + float(phone["duration"]) for phone in group)
        if group_start - cursor >= 0.005:
            events.append(("SP", group_start - cursor))
        next_start = float(groups[index + 1][0]["start"]) if index + 1 < len(groups) else crop_end
        blank = max(0.0, next_start - group_end)
        if index + 1 < len(groups) and blank > MAX_INTERSYLLABLE_BLANK_SECONDS:
            voiced_end = min(crop_end, group_end)
            silence_end = min(crop_end, next_start)
        else:
            voiced_end = min(crop_end, next_start)
            silence_end = voiced_end
        duration = max(0.02, voiced_end - group_start)
        symbols = [phone["symbol"] for phone in group]
        durations = split_phone_durations(symbols, duration)
        events.extend(zip(symbols, durations))
        cursor = group_start + sum(durations)
        if silence_end - cursor >= 0.005:
            events.append(("SP", silence_end - cursor))
            cursor = silence_end
    if crop_end - cursor >= 0.005:
        events.append(("SP", crop_end - cursor))
    drift = crop_end - crop_start - sum(duration for _, duration in events)
    events[-1] = (events[-1][0], events[-1][1] + drift)
    return [symbol for symbol, _ in events], [duration for _, duration in events]


def write_teacher(teacher: str, rows: list[dict]) -> tuple[dict, list[dict]]:
    raw = WORK / f"raw/ko_teacher_coarticulated_{teacher}"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    output = []
    transcription = []
    for row in rows:
        source = ROOT / row["audio_path"]
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        duration = len(audio) / sample_rate
        groups = syllable_groups(timed_phones(row))
        first = groups[0][0]
        last = groups[-1][-1]
        crop_start = max(0.0, float(first["start"]) - LEADING_CONTEXT_SECONDS)
        crop_end = min(duration, float(last["start"]) + float(last["duration"]) + TRAILING_CONTEXT_SECONDS)
        phones, durations = labels(groups, crop_start, crop_end)
        start_frame, end_frame = round(crop_start * sample_rate), round(crop_end * sample_rate)
        target = wavs / f"{row['id']}.wav"
        sf.write(target, audio[start_frame:end_frame], sample_rate, subtype="PCM_24")
        actual_duration = (end_frame - start_frame) / sample_rate
        durations[-1] += actual_duration - sum(durations)
        item = {
            "id": f"{teacher}_{row['id']}",
            "source_id": row["id"],
            "teacher": row["teacher"],
            "text": row["text"],
            "audio_path": str(target.relative_to(ROOT)),
            "duration_seconds": actual_duration,
            "ph_seq": phones,
            "ph_dur": durations,
            "split": "validation" if int(row["id"].rsplit("_", 1)[1]) > 25 else "train",
            "label_status": "inferred_ctc_anchor_with_speech_coarticulation_prior",
            "training_role": "low_trust_korean_teacher_lexical_prior_not_real_gyu_singing",
        }
        output.append(item)
        transcription.append({
            "name": row["id"], "ph_seq": " ".join(phones),
            "ph_dur": " ".join(f"{value:.6f}" for value in durations),
        })
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(transcription)
    return ({
        "raw_data_dir": str(raw), "speaker": "gyu", "language": "gyu", "spk_id": 20,
        "test_prefixes": [row["source_id"] for row in output if row["split"] == "validation"],
    }, output)


def pause_ratio(rows: list[dict]) -> float:
    total = sum(sum(map(float, row["ph_dur"])) for row in rows)
    pauses = sum(
        float(duration) for row in rows
        for symbol, duration in zip(row["ph_seq"], row["ph_dur"])
        if symbol in {"SP", "AP"}
    )
    return pauses / total


def main() -> None:
    source = [row for row in read_jsonl(SOURCE) if row["accepted"]]
    datasets = []
    output = []
    for teacher in ("fish", "moss"):
        dataset, rows = write_teacher(
            teacher, [row for row in source if row["teacher"].startswith(teacher)]
        )
        datasets.append(dataset)
        output.extend(rows)
    # DiffSinger requires AP to occur; use one boundary pause without changing duration.
    first_pause = next(
        (row, index) for row in output for index, symbol in enumerate(row["ph_seq"])
        if symbol == "SP"
    )
    first_pause[0]["ph_seq"][first_pause[1]] = "AP"
    raw_path = Path(datasets[0]["raw_data_dir"]) / "transcriptions.csv"
    lines = list(csv.DictReader(raw_path.open()))
    lines[0]["ph_seq"] = " ".join(output[0]["ph_seq"])
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader(); writer.writerows(lines)

    manifest = ROOT / "data/manifests/diffsinger_teacher_coarticulated.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    gyu_rows = read_jsonl(GYU)
    phonemes = {
        symbol for row in output + gyu_rows for symbol in row["ph_seq"]
        if symbol not in {"SP", "AP"}
    }
    dictionary = WORK / "dictionary-gyu-teacher-coarticulated.txt"
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phonemes)))
    checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_teacher_coarticulated_vocab.ckpt"
    vocabulary = remap_vocabulary(
        ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000.ckpt",
        WORK / "dictionary-gyu.txt", dictionary, checkpoint,
    )
    config = yaml.safe_load((ROOT / "configs/diffsinger_gyu_coarticulated.yaml").read_text())
    config["datasets"].extend(datasets)
    config["dictionaries"]["gyu"] = str(dictionary)
    config["binary_data_dir"] = str(WORK / "binary_gyu_teacher_coarticulated")
    config["finetune_ckpt_path"] = str(checkpoint)
    # One bounded continuation checks whether the corrected Korean lexical prior
    # emerges after the first 600-step acoustic-only failure. Do not extend this
    # branch again unless stress ASR improves.
    config["max_updates"] = 1200
    config["val_check_interval"] = 300
    config["num_ckpt_keep"] = 6
    config["optimizer_args"]["lr"] = 1e-5
    config_path = ROOT / "configs/diffsinger_teacher_coarticulated.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "coarticulated_teacher_prior_ready",
        "teacher_rows": len(output),
        "unique_texts": len({row["text"] for row in output}),
        "teacher_distribution": {
            teacher: sum(row["teacher"].startswith(teacher) for row in output)
            for teacher in ("fish", "moss")
        },
        "old_teacher_pause_ratio": round(pause_ratio(source), 6),
        "new_teacher_pause_ratio": round(pause_ratio(output), 6),
        "real_gyu_rows": len(gyu_rows),
        "independent_score_rows_included": False,
        "teacher_speech_is_real_gyu_singing": False,
        "maximum_bridged_ctc_blank_seconds": MAX_INTERSYLLABLE_BLANK_SECONDS,
        "label_status": "inferred",
        "maximum_training_steps": 1200,
        "continuation_gate": "stop unless rapid and interval stress ASR improves",
        "manifest": str(manifest.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "vocabulary_remap": vocabulary,
    }
    target = ROOT / "artifacts/reports/diffsinger_teacher_coarticulated.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
