#!/usr/bin/env python3
"""Build full-phrase GYU rows without CTC blank gaps between sung syllables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf
import yaml

from prepare_diffsinger_gyu_segments import remap_vocabulary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
RAW = WORK / "raw/gyu_coarticulated"
MAX_BRIDGED_GAP_SECONDS = 0.8
LEADING_CONTEXT_SECONDS = 0.02
TRAILING_CONTEXT_SECONDS = 0.03


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def syllable_groups(phones: list[dict]) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for phone in phones:
        grouped.setdefault(int(phone["syllable_index"]), []).append(phone)
    return [grouped[index] for index in sorted(grouped)]


def split_phone_durations(symbols: list[str], duration: float) -> list[float]:
    if len(symbols) == 1:
        return [duration]
    onset = min(0.08, max(0.018, duration * 0.18))
    coda = min(0.10, max(0.018, duration * 0.18)) if len(symbols) == 3 else 0.0
    nucleus = duration - onset - coda
    if nucleus < 0.02:
        scale = max(0.0, duration - 0.02) / max(onset + coda, 1e-8)
        onset *= scale
        coda *= scale
        nucleus = duration - onset - coda
    return [onset, nucleus] if len(symbols) == 2 else [onset, nucleus, coda]


def coarticulated_labels(groups: list[list[dict]], crop_start: float, crop_end: float) -> tuple[list[str], list[float]]:
    events: list[tuple[str, float]] = []
    cursor = crop_start
    for index, group in enumerate(groups):
        group_start = max(cursor, min(float(phone["start"]) for phone in group))
        group_end = max(float(phone["start"]) + float(phone["duration"]) for phone in group)
        if group_start - cursor >= 0.005:
            events.append(("SP", group_start - cursor))
        next_start = (
            min(float(phone["start"]) for phone in groups[index + 1])
            if index + 1 < len(groups)
            else crop_end
        )
        blank_gap = max(0.0, next_start - group_end)
        if index + 1 < len(groups) and blank_gap > MAX_BRIDGED_GAP_SECONDS:
            sung_end = min(crop_end, group_end)
            silence_end = min(crop_end, next_start)
        else:
            sung_end = min(crop_end, next_start)
            silence_end = sung_end
        duration = max(0.02, sung_end - group_start)
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
    if events[-1][1] <= 0:
        raise RuntimeError("non-positive final label duration")
    return [symbol for symbol, _ in events], [duration for _, duration in events]


def main() -> None:
    source_rows = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data/manifests/diffsinger_score_native.jsonl")
        if row["training_allowed"]
    }
    alignments = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data/manifests/real_phoneme_alignment.jsonl")
        if row["id"] in source_rows
    }
    independent = {
        row["id"] for row in read_jsonl(ROOT / "data/manifests/manual_verified_scores.jsonl")
    }
    if set(source_rows) & independent:
        raise RuntimeError("independent-score rows leaked into coarticulation training")

    wavs = RAW / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    output: list[dict] = []
    transcriptions: list[dict] = []
    for identifier, source in sorted(source_rows.items()):
        groups = syllable_groups(alignments[identifier]["phones"])
        if len(groups) != len(source["score_notes"]):
            raise RuntimeError(f"syllable/note mismatch: {identifier}")
        audio, sample_rate = sf.read(ROOT / source["audio_path"], dtype="float32", always_2d=True)
        source_duration = len(audio) / sample_rate
        crop_start = max(0.0, min(float(phone["start"]) for phone in groups[0]) - LEADING_CONTEXT_SECONDS)
        last_end = max(float(phone["start"]) + float(phone["duration"]) for phone in groups[-1])
        crop_end = min(source_duration, last_end + TRAILING_CONTEXT_SECONDS)
        phones, durations = coarticulated_labels(groups, crop_start, crop_end)
        first, last = round(crop_start * sample_rate), round(crop_end * sample_rate)
        target = wavs / f"{identifier}.wav"
        sf.write(target, audio[first:last], sample_rate, subtype="PCM_24")
        actual_duration = (last - first) / sample_rate
        durations[-1] += actual_duration - sum(durations)
        row = {
            "id": identifier,
            "source_id": identifier,
            "audio_path": str(target.relative_to(ROOT)),
            "duration_seconds": actual_duration,
            "source_start_seconds": first / sample_rate,
            "source_end_seconds": last / sample_rate,
            "ph_seq": phones,
            "ph_dur": durations,
            "split": source["split"],
            "label_status": "inferred_ctc_anchor_with_singing_coarticulation_prior",
            "training_role": "real_gyu_phrase_score_native_acoustic",
            "source_recording_modified": False,
        }
        output.append(row)
        transcriptions.append({
            "name": identifier,
            "ph_seq": " ".join(phones),
            "ph_dur": " ".join(f"{value:.6f}" for value in durations),
        })

    keep = {f"{row['id']}.wav" for row in output}
    for stale in wavs.glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    with (RAW / "transcriptions.csv").open("w", newline="") as handle:
        # DiffSinger requires both pause symbols to occur at least once.
        first_pause = output[0]["ph_seq"].index("SP")
        output[0]["ph_seq"][first_pause] = "AP"
        transcriptions[0]["ph_seq"] = " ".join(output[0]["ph_seq"])
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(transcriptions)

    manifest = ROOT / "data/manifests/diffsinger_gyu_coarticulated.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))

    old_dictionary = WORK / "dictionary-gyu.txt"
    dictionary = WORK / "dictionary-gyu-coarticulated.txt"
    phonemes = {phone for row in output for phone in row["ph_seq"]} - {"AP", "SP"}
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phonemes)))
    source_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000.ckpt"
    target_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_coarticulated_vocab.ckpt"
    vocabulary = remap_vocabulary(source_checkpoint, old_dictionary, dictionary, target_checkpoint)

    config = yaml.safe_load((ROOT / "configs/diffsinger_gyu_all_adapt.yaml").read_text())
    config["datasets"] = [{
        "raw_data_dir": str(RAW),
        "speaker": "gyu",
        "language": "gyu",
        "spk_id": 20,
        "test_prefixes": [row["id"] for row in output if row["split"] != "train"],
    }]
    config["dictionaries"]["gyu"] = str(dictionary)
    config["binary_data_dir"] = str(WORK / "binary_gyu_coarticulated")
    config["finetune_ckpt_path"] = str(target_checkpoint)
    config["max_updates"] = 600
    config["val_check_interval"] = 100
    config["optimizer_args"]["lr"] = 1e-5
    config_path = ROOT / "configs/diffsinger_gyu_coarticulated.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    original_uncovered = []
    for row in source_rows.values():
        original_uncovered.append(1.0 - sum(row["ph_dur"]) / row["duration_seconds"])
    previous_sp = previous_total = 0.0
    with (WORK / "raw/gyu/transcriptions.csv").open() as handle:
        for row in csv.DictReader(handle):
            for symbol, duration in zip(row["ph_seq"].split(), map(float, row["ph_dur"].split())):
                previous_total += duration
                if symbol == "SP":
                    previous_sp += duration
    internal_sp = sum(
        duration
        for row in output
        for symbol, duration in zip(row["ph_seq"], row["ph_dur"])
        if symbol == "SP"
    )
    total = sum(row["duration_seconds"] for row in output)
    report = {
        "status": "coarticulated_training_rows_ready",
        "rows": len(output),
        "train_rows": sum(row["split"] == "train" for row in output),
        "validation_rows": sum(row["split"] != "train" for row in output),
        "duration_minutes": round(total / 60, 3),
        "independent_score_rows_included": False,
        "source_recordings_modified": False,
        "label_status": "inferred",
        "previous_manifest_uncovered_duration_ratio_mean": round(
            sum(original_uncovered) / len(original_uncovered), 6
        ),
        "previous_binarized_sp_duration_ratio": round(previous_sp / previous_total, 6),
        "new_sp_duration_ratio": round(internal_sp / total, 6),
        "maximum_bridged_ctc_blank_seconds": MAX_BRIDGED_GAP_SECONDS,
        "manifest": str(manifest.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "vocabulary_remap": vocabulary,
    }
    (ROOT / "artifacts/reports/diffsinger_gyu_coarticulated.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
