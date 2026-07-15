#!/usr/bin/env python3
"""Stream a bounded Zeroth-Korean phoneme/acoustic prior for DiffSinger."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from datasets import Audio, load_dataset

from gyu_singer.frontend import phonemize
from gyu_singer.inference.content_timing import roman_phone


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
RAW = WORK / "raw"
REVISION = "1fe937899f828af822293d05e086200946088bdf"
SPEAKER_IDS = (187, 191, 201, 214)
ROWS_PER_SPEAKER = 100
VALID_PER_SPEAKER = 20
MIN_CTC_LOG_SCORE = -3.0


def phone_alignment(text: str, audio: np.ndarray, model, labels, device: str) -> dict:
    text = "".join(re.findall(r"[가-힣]", text))
    phone_tokens = [
        (symbol, token)
        for symbol in phonemize("ko", text).symbols
        if (token := roman_phone(symbol))
    ]
    characters = "".join(token for _, token in phone_tokens)
    dictionary = {label: index for index, label in enumerate(labels)}
    target = torch.tensor([[dictionary[char] for char in characters]], device=device)
    with torch.inference_mode():
        emission, _ = model(torch.from_numpy(audio)[None].to(device))
    alignment, scores = torchaudio.functional.forced_align(emission.log_softmax(-1), target)
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    seconds = len(audio) / 16_000 / emission.shape[1]

    phones, span_index = [], 0
    for symbol, token in phone_tokens:
        group = spans[span_index:span_index + len(token)]
        span_index += len(token)
        if len(group) != len(token):
            raise RuntimeError("incomplete CTC alignment")
        phones.append({
            "symbol": symbol,
            "start": group[0].start * seconds,
            "end": group[-1].end * seconds,
            "score": float(np.mean([span.score for span in group])),
        })

    duration, cursor, events = len(audio) / 16_000, 0.0, []
    for phone in phones:
        start = max(cursor, phone["start"])
        if start - cursor >= .005:
            events.append(("SP", start - cursor))
        else:
            start = cursor
        end = min(duration, max(start + .005, phone["end"]))
        events.append((phone["symbol"], end - start))
        cursor = end
    if duration > cursor:
        events.append(("SP", duration - cursor))
    return {
        "ph_seq": [symbol for symbol, _ in events],
        "ph_dur": [length for _, length in events],
        "ctc_mean_log_score": float(np.mean([phone["score"] for phone in phones])),
    }


def write_raw(
    speaker: int,
    rows: list[dict],
    speaker_index: int,
    label: str,
    valid_per_speaker: int,
) -> dict:
    raw = RAW / f"{label}_{speaker}"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "name": row["id"],
                "ph_seq": " ".join(row["ph_seq"]),
                "ph_dur": " ".join(f"{value:.6f}" for value in row["ph_dur"]),
            })
    return {
        "raw_data_dir": str(raw),
        "speaker": f"{label}_{speaker}",
        "spk_id": 21 + speaker_index,
        "language": "gyu",
        "test_prefixes": [row["id"] for row in rows[-valid_per_speaker:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-speaker", type=int, default=ROWS_PER_SPEAKER)
    parser.add_argument("--valid-per-speaker", type=int, default=VALID_PER_SPEAKER)
    parser.add_argument("--label", default="zeroth")
    args = parser.parse_args()
    if not 0 < args.valid_per_speaker < args.rows_per_speaker:
        parser.error("valid-per-speaker must be between zero and rows-per-speaker")
    if not re.fullmatch(r"[a-z0-9_]+", args.label):
        parser.error("label must contain only lowercase letters, digits, and underscores")
    os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    labels, model = bundle.get_labels(), bundle.get_model().to(device).eval()
    stream = load_dataset(
        "kresnik/zeroth_korean", split="train", streaming=True, revision=REVISION
    ).cast_column("audio", Audio(decode=False))

    selected: dict[int, list[dict]] = defaultdict(list)
    speaker_order = list(SPEAKER_IDS)
    for source in stream:
        speaker = int(source["speaker_id"])
        if speaker not in SPEAKER_IDS:
            continue
        if len(selected[speaker]) >= args.rows_per_speaker:
            if all(len(selected[value]) >= args.rows_per_speaker for value in speaker_order):
                break
            continue
        text = source["text"]
        if len(re.findall(r"[가-힣]", text)) < 5:
            continue
        audio, rate = sf.read(io.BytesIO(source["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != 16_000:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, 16_000).numpy()
        try:
            aligned = phone_alignment(text, audio.astype("float32"), model, labels, device)
        except (RuntimeError, KeyError):
            continue
        if aligned["ctc_mean_log_score"] < MIN_CTC_LOG_SCORE:
            continue
        identifier = f"{args.label}_{source['id']}"
        wav = RAW / f"{args.label}_{speaker}" / "wavs" / f"{identifier}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav, audio, 16_000, subtype="PCM_16")
        selected[speaker].append({
            "id": identifier,
            "source_id": source["id"],
            "speaker_id": speaker,
            "text": text,
            "audio_path": str(wav.relative_to(ROOT)),
            "duration_seconds": len(audio) / 16_000,
            "split": "validation"
            if len(selected[speaker]) >= args.rows_per_speaker - args.valid_per_speaker
            else "train",
            "label_status": "inferred_mms_ctc",
            "training_role": "korean_phoneme_acoustic_prior_only_not_singing",
            **aligned,
        })
        if all(len(selected[value]) >= args.rows_per_speaker for value in speaker_order):
            break

    if any(len(selected[value]) != args.rows_per_speaker for value in speaker_order):
        raise RuntimeError("stream ended before the bounded speaker-balanced subset was complete")
    rows = [row for speaker in speaker_order for row in selected[speaker]]
    for index, symbol in enumerate(rows[0]["ph_seq"]):
        if symbol == "SP":
            rows[0]["ph_seq"][index] = "AP"
            break
    manifest = ROOT / f"data/manifests/diffsinger_{args.label}_prior.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    base = yaml.safe_load((ROOT / "configs/diffsinger_score_native.yaml").read_text())
    text = yaml.safe_load((ROOT / "configs/diffsinger_score_native.yaml").read_text())
    old_dictionary = Path(base["dictionaries"]["gyu"])
    old = {line.split("\t", 1)[0] for line in old_dictionary.read_text().splitlines() if line}
    phones = {phone for row in rows for phone in row["ph_seq"]} | old
    dictionary = WORK / f"dictionary-gyu-{args.label}.txt"
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP"})))
    text_dictionary = WORK / f"dictionary-gyu-{args.label}-text.txt"
    text_dictionary.write_text("".join(
        f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP", "a", "e", "i", "o", "u"})
    ))
    zeroth = [
        write_raw(
            speaker,
            selected[speaker],
            index,
            args.label,
            args.valid_per_speaker,
        )
        for index, speaker in enumerate(speaker_order)
    ]
    gyu = base["datasets"][-1]
    gyu["spk_id"] = 20
    base["datasets"] += zeroth
    base["dictionaries"]["gyu"] = str(dictionary)
    base["num_spk"] = 21 + len(SPEAKER_IDS)
    base["binary_data_dir"] = str(WORK / f"binary_{args.label}_gyu")
    config = ROOT / f"configs/diffsinger_{args.label}_prior.yaml"
    config.write_text(yaml.safe_dump(base, sort_keys=False))

    text["datasets"][-1]["spk_id"] = 20
    text["datasets"] = [text["datasets"][-1], *zeroth]
    text["dictionaries"]["gyu"] = str(text_dictionary)
    text["num_spk"] = 21 + len(SPEAKER_IDS)
    text["binary_data_dir"] = str(WORK / f"binary_{args.label}_text")
    text_config = ROOT / f"configs/diffsinger_{args.label}_text_prior.yaml"
    text_config.write_text(yaml.safe_dump(text, sort_keys=False))

    report = {
        "status": "bounded_prior_ready_for_binarization",
        "dataset": "Zeroth-Korean SLR40",
        "license": "CC BY 4.0",
        "official_source": "https://www.openslr.org/40/",
        "streaming_mirror": "kresnik/zeroth_korean",
        "mirror_revision": REVISION,
        "rows": len(rows),
        "speakers": speaker_order,
        "rows_per_speaker": args.rows_per_speaker,
        "train_rows": sum(row["split"] == "train" for row in rows),
        "validation_rows": sum(row["split"] == "validation" for row in rows),
        "hours": round(sum(row["duration_seconds"] for row in rows) / 3600, 3),
        "ctc_score_mean": round(float(np.mean([row["ctc_mean_log_score"] for row in rows])), 4),
        "training_role": "speech phoneme/acoustic prior only; never real or pseudo singing",
        "full_corpus_downloaded": False,
        "acoustic_replay_config": str(config.relative_to(ROOT)),
        "text_only_config": str(text_config.relative_to(ROOT)),
    }
    output = ROOT / f"artifacts/reports/diffsinger_{args.label}_prior.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    os._exit(0)  # datasets 4.8 AudioDecoder can abort during interpreter teardown


if __name__ == "__main__":
    main()
