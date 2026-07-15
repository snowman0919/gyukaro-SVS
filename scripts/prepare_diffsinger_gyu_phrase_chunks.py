#!/usr/bin/env python3
"""Build multi-syllable real-GYU phrase chunks for coarticulation training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf
import yaml

from prepare_diffsinger_gyu_segments import remap_vocabulary, segment_labels, sha256


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
GAP_SECONDS = 0.8
MAX_CHUNK_SECONDS = 6.0
MIN_CHUNK_SECONDS = 0.6


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def groups(phones: list[dict]) -> list[list[dict]]:
    output: list[list[dict]] = []
    current: list[dict] = []
    cursor = 0.0
    for phone in phones:
        start = float(phone["start"])
        elapsed = start - float(current[0]["start"]) if current else 0.0
        boundary = (
            current
            and (
                start - cursor >= GAP_SECONDS
                or (
                    elapsed >= MAX_CHUNK_SECONDS
                    and phone["symbol"].startswith("ko_onset_")
                )
            )
        )
        if boundary:
            output.append(current)
            current = []
        current.append(phone)
        cursor = start + float(phone["duration"])
    if current:
        output.append(current)
    return output


def main() -> None:
    sources = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data/manifests/real_segments.jsonl")
        if row["segment_type"] == "phrase" and row["source_index"] >= 146
    }
    recordings = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data/manifests/real_recordings.jsonl")
    }
    score_splits = {
        row["id"]: row["split"]
        for row in read_jsonl(ROOT / "data/manifests/diffsinger_score_native.jsonl")
    }
    alignments = {
        row["id"]: row["phones"]
        for row in read_jsonl(ROOT / "data/manifests/real_phoneme_alignment_all.jsonl")
        if row["id"] in sources
    }
    if set(alignments) & {f"gyu_real_{index:06d}" for index in range(171, 195)}:
        raise RuntimeError("independent-score evaluation rows leaked into phrase chunks")

    raw = WORK / "raw/gyu_phrase_chunks"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    transcription_rows: list[dict] = []
    for source_id, phones in sorted(alignments.items()):
        recording = recordings[source_id]
        source_audio = ROOT / recording["pcm_master"]
        audio, sample_rate = sf.read(source_audio, dtype="float32", always_2d=True)
        for index, group in enumerate(groups(phones)):
            start = max(0.0, float(group[0]["start"]) - 0.02)
            end = min(
                len(audio) / sample_rate,
                float(group[-1]["start"]) + float(group[-1]["duration"]) + 0.03,
            )
            if end - start < MIN_CHUNK_SECONDS:
                continue
            ph_seq, ph_dur = segment_labels(group, start, end)
            identifier = f"{source_id}_phrase{index:02d}"
            target = wavs / f"{identifier}.wav"
            first, last = round(start * sample_rate), round(end * sample_rate)
            sf.write(target, audio[first:last], sample_rate, subtype="PCM_24")
            duration = (last - first) / sample_rate
            ph_dur[-1] += duration - sum(ph_dur)
            split = score_splits.get(
                source_id,
                "validation" if recording["source_index"] % 5 == 0 else "train",
            )
            row = {
                "id": identifier,
                "source_id": source_id,
                "source_text": sources[source_id]["text"],
                "audio_path": str(target.relative_to(ROOT)),
                "source_audio_sha256": sha256(source_audio),
                "sample_rate": sample_rate,
                "duration_seconds": duration,
                "source_start_seconds": start,
                "source_end_seconds": end,
                "ph_seq": ph_seq,
                "ph_dur": ph_dur,
                "split": split,
                "script_block": sources[source_id]["script_block"],
                "label_status": "inferred_from_singing_aware_alignment",
                "training_role": "real_gyu_multisyllable_phrase_coarticulation",
                "source_recording_modified": False,
            }
            rows.append(row)
            transcription_rows.append({
                "name": identifier,
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(f"{value:.6f}" for value in ph_dur),
            })

    for row, transcription in zip(rows, transcription_rows):
        if "SP" in row["ph_seq"]:
            index = row["ph_seq"].index("SP")
            row["ph_seq"][index] = "AP"
            transcription["ph_seq"] = " ".join(row["ph_seq"])
            break
    keep = {f'{row["id"]}.wav' for row in rows}
    for stale in wavs.glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(transcription_rows)

    manifest = ROOT / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    config = yaml.safe_load((ROOT / "configs/diffsinger_gyu_all_adapt.yaml").read_text())
    old_dictionary = Path(config["dictionaries"]["gyu"])
    dictionary = WORK / "dictionary-gyu-phrase-chunks.txt"
    phonemes = {phone for row in rows for phone in row["ph_seq"]}
    dictionary.write_text("".join(
        f"{phone}\t{phone}\n" for phone in sorted(phonemes - {"AP", "SP"})
    ))
    target_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_gyu_phrase_vocab.ckpt"
    remap = remap_vocabulary(
        Path(config["finetune_ckpt_path"]), old_dictionary, dictionary, target_checkpoint
    )
    config["datasets"] = [{
        "raw_data_dir": str(raw),
        "speaker": "gyu",
        "language": "gyu",
        "spk_id": 20,
        "test_prefixes": [row["id"] for row in rows if row["split"] != "train"],
    }]
    config["dictionaries"]["gyu"] = str(dictionary)
    config["binary_data_dir"] = str(WORK / "binary_gyu_phrase_chunks")
    config["finetune_ckpt_path"] = str(target_checkpoint)
    config["max_updates"] = 300
    config["val_check_interval"] = 100
    config["optimizer_args"]["lr"] = 5e-6
    output_config = ROOT / "configs/diffsinger_gyu_phrase_chunks.yaml"
    output_config.write_text(yaml.safe_dump(config, sort_keys=False))

    durations = [row["duration_seconds"] for row in rows]
    report = {
        "status": "real_gyu_phrase_chunks_ready",
        "source_rows": len(alignments),
        "chunks": len(rows),
        "train_chunks": sum(row["split"] == "train" for row in rows),
        "validation_chunks": sum(row["split"] != "train" for row in rows),
        "duration_minutes": round(sum(durations) / 60, 3),
        "median_chunk_seconds": round(sorted(durations)[len(durations) // 2], 3),
        "subsecond_chunks": sum(value < 1.0 for value in durations),
        "gap_split_seconds": GAP_SECONDS,
        "max_chunk_seconds": MAX_CHUNK_SECONDS,
        "minimum_chunk_seconds": MIN_CHUNK_SECONDS,
        "independent_score_rows_included": False,
        "labels": "inferred singing-aware CTC timing",
        "source_recordings_modified": False,
        "vocabulary_remap": remap,
        "manifest": str(manifest.relative_to(ROOT)),
        "config": str(output_config.relative_to(ROOT)),
    }
    (ROOT / "artifacts/reports/diffsinger_gyu_phrase_chunks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
