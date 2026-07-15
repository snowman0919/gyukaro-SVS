#!/usr/bin/env python3
"""Add low-trust Korean teacher speech as a phoneme prior for DiffSinger."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from scipy.signal import resample_poly

from gyu_singer.frontend import phonemize
from gyu_singer.inference.content_timing import roman_phone


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCES = {
    "fish": ROOT / "data/manifests/teacher_v06_unique_fish.jsonl",
    "moss": ROOT / "data/manifests/teacher_v06_unique_moss.jsonl",
}
MIN_CTC_LOG_SCORE = -3.0


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def align(row: dict, model: torch.nn.Module, labels: tuple[str, ...], device: str) -> dict:
    text = "".join(re.findall(r"[가-힣]", row["text"]))
    phone_tokens = [
        (symbol, token)
        for symbol in phonemize("ko", text).symbols
        if (token := roman_phone(symbol))
    ]
    characters = "".join(token for _, token in phone_tokens)
    dictionary = {label: index for index, label in enumerate(labels)}
    if missing := sorted(set(characters) - dictionary.keys()):
        raise ValueError(f"unsupported MMS characters: {missing}")

    path = ROOT / row["output_path"]
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    duration = len(audio) / rate
    aligned = resample_poly(audio, 16_000, rate).astype("float32") if rate != 16_000 else audio
    with torch.inference_mode():
        emission, _ = model(torch.from_numpy(aligned)[None].to(device))
    target = torch.tensor([[dictionary[char] for char in characters]], device=device)
    alignment, scores = torchaudio.functional.forced_align(emission.log_softmax(-1), target)
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    seconds = duration / emission.shape[1]

    phones, cursor = [], 0
    for symbol, token in phone_tokens:
        group = spans[cursor:cursor + len(token)]
        cursor += len(token)
        if len(group) != len(token):
            raise RuntimeError(f"incomplete CTC alignment: {row['id']}")
        phones.append({
            "symbol": symbol,
            "start": group[0].start * seconds,
            "end": group[-1].end * seconds,
            "ctc_log_score": float(np.mean([span.score for span in group])),
        })

    events, cursor_time = [], 0.0
    for phone in phones:
        start = max(cursor_time, phone["start"])
        if start - cursor_time >= .005:
            events.append(("SP", start - cursor_time))
        else:
            start = cursor_time
        end = min(duration, max(start + .005, phone["end"]))
        events.append((phone["symbol"], end - start))
        cursor_time = end
    if duration > cursor_time:
        events.append(("SP", duration - cursor_time))
    score = float(np.mean([phone["ctc_log_score"] for phone in phones]))
    return {
        "id": row["id"],
        "teacher": row["teacher"],
        "text": row["text"],
        "audio_path": row["output_path"],
        "duration_seconds": duration,
        "ph_seq": [symbol for symbol, _ in events],
        "ph_dur": [length for _, length in events],
        "ctc_mean_log_score": score,
        "accepted": score >= MIN_CTC_LOG_SCORE,
        "label_status": "inferred_mms_ctc_low_trust_teacher_speech",
        "training_role": "korean_phoneme_content_prior_only_not_real_gyu_singing",
    }


def write_raw(teacher: str, rows: list[dict]) -> dict:
    raw = WORK / "raw" / f"ko_phoneme_prior_{teacher}"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        for row in rows:
            target = wavs / f"{row['id']}.wav"
            if not target.exists():
                target.symlink_to(ROOT / row["audio_path"])
            writer.writerow({
                "name": row["id"],
                "ph_seq": " ".join(row["ph_seq"]),
                "ph_dur": " ".join(f"{value:.6f}" for value in row["ph_dur"]),
            })
    validation = [row["id"] for row in rows if int(row["id"].rsplit("_", 1)[1]) > 25]
    return {
        "raw_data_dir": str(raw), "speaker": "gyu", "spk_id": 20,
        "language": "gyu", "test_prefixes": validation,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    labels, model = bundle.get_labels(), bundle.get_model().to(device).eval()
    aligned = []
    for teacher, source in SOURCES.items():
        for row in read(source):
            if row["language"] == "ko":
                aligned.append(align(row, model, labels, device))

    accepted = [row for row in aligned if row["accepted"]]
    for index, symbol in enumerate(accepted[0]["ph_seq"]):
        if symbol == "SP":
            accepted[0]["ph_seq"][index] = "AP"
            break
    manifest = ROOT / "data/manifests/diffsinger_ko_phoneme_prior.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned))

    config = yaml.safe_load((ROOT / "configs/diffsinger_score_native.yaml").read_text())
    old_dictionary = Path(config["dictionaries"]["gyu"])
    dictionary = WORK / "dictionary-gyu-ko-prior.txt"
    old_lines = [line for line in old_dictionary.read_text().splitlines() if line]
    removed = {"a", "e", "i", "o", "u"}
    existing = [line.split("\t", 1)[0] for line in old_lines if line.split("\t", 1)[0] not in removed]
    added = sorted({phone for row in accepted for phone in row["ph_seq"] if phone != "SP"} - set(existing))
    dictionary.write_text("".join(f"{line}\n" for line in old_lines if line.split("\t", 1)[0] not in removed) + "".join(f"{phone}\t{phone}\n" for phone in added if phone != "AP"))
    config["dictionaries"]["gyu"] = str(dictionary)
    config["datasets"][-1]["spk_id"] = 20
    config["datasets"] = [config["datasets"][-1]]
    for teacher in SOURCES:
        config["datasets"].append(write_raw(teacher, [row for row in accepted if row["teacher"].startswith(teacher)]))
    config["binary_data_dir"] = str(WORK / "binary_ko_prior_gyu")
    target_config = ROOT / "configs/diffsinger_ko_phoneme_prior.yaml"
    target_config.write_text(yaml.safe_dump(config, sort_keys=False))

    report = {
        "status": "inferred_prior_ready_for_binarization",
        "candidates": len(aligned), "accepted": len(accepted),
        "teacher_rows": {teacher: sum(row["teacher"].startswith(teacher) and row["accepted"] for row in aligned) for teacher in SOURCES},
        "ctc_threshold": MIN_CTC_LOG_SCORE,
        "ctc_score_mean": round(float(np.mean([row["ctc_mean_log_score"] for row in accepted])), 4),
        "phonemes_added": [phone for phone in added if phone != "AP"],
        "vocalset_only_phonemes_removed": sorted(removed),
        "training_role": "phoneme content prior only; teacher speech is not real GYU singing",
        "speaker_id_policy": "GYU-reference teacher speech shares frozen GYU conditioning; singing decoder adaptation remains separate",
        "semantic_split": "IDs 026-030 held out for both teachers",
        "audio_copy": False,
        "config": str(target_config.relative_to(ROOT)),
    }
    output = ROOT / "artifacts/reports/diffsinger_ko_phoneme_prior.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
