#!/usr/bin/env python3
"""Prepare the CC-BY VocalSet Row Your Boat lexical-singing pilot."""
from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from scipy.signal import resample_poly

from gyu_singer.frontend import phonemize
from prepare_diffsinger_gyu_segments import remap_vocabulary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
ARCHIVE = ROOT / "data/external/work/VocalSet1-2.fixed.zip"
INDEX_ARCHIVE = ARCHIVE
WORDS = "row row row your boat gently down the stream merrily merrily merrily merrily life is but a dream".split()
VALIDATION_SPEAKERS = {"female2", "female9", "male7", "male10"}
VOWELS = {
    "en_aa", "en_ae", "en_ah", "en_ao", "en_aw", "en_ay", "en_eh", "en_er",
    "en_ey", "en_ih", "en_iy", "en_ow", "en_oy", "en_uh", "en_uw",
}


def align_words(audio: np.ndarray, rate: int, model, labels: tuple[str, ...], device: str) -> tuple[list[dict], float]:
    analysis = resample_poly(audio, 16_000, rate).astype("float32") if rate != 16_000 else audio
    target = "".join(WORDS)
    dictionary = {label: index for index, label in enumerate(labels)}
    tokens = torch.tensor([[dictionary[char] for char in target]], device=device)
    with torch.inference_mode():
        emission, _ = model(torch.from_numpy(analysis)[None].to(device))
    alignment, scores = torchaudio.functional.forced_align(emission.log_softmax(-1), tokens)
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    if len(spans) != len(target):
        raise RuntimeError("incomplete VocalSet character alignment")
    seconds = len(analysis) / 16_000 / emission.shape[1]
    character_spans = [(span.start * seconds, span.end * seconds) for span in spans]
    output, cursor = [], 0
    for word in WORDS:
        group = character_spans[cursor:cursor + len(word)]
        cursor += len(word)
        phones = phonemize("en", word).symbols
        start, end = group[0][0], group[-1][1]
        weights = np.array([3.0 if phone in VOWELS else 1.0 for phone in phones])
        durations = (end - start) * weights / weights.sum()
        output.append({"word": word, "start": start, "end": end, "phones": phones, "durations": durations.tolist()})
    return output, float(scores[0].mean())


def phrase_labels(words: list[dict], start: float, end: float) -> tuple[list[str], list[float]]:
    events, cursor = [], start
    for word in words:
        if word["start"] - cursor >= 0.005:
            events.append(("SP", word["start"] - cursor))
        phone_start = word["start"]
        for phone, duration in zip(word["phones"], word["durations"]):
            events.append((phone, duration))
            phone_start += duration
        cursor = word["end"]
    if end - cursor >= 0.005:
        events.append(("SP", end - cursor))
    drift = end - start - sum(duration for _, duration in events)
    events[-1] = (events[-1][0], events[-1][1] + drift)
    return [phone for phone, _ in events], [duration for _, duration in events]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    labels, model = bundle.get_labels(), bundle.get_model().to(device).eval()
    base = yaml.safe_load((ROOT / "configs/diffsinger_gyu_all_segmented_finetune.yaml").read_text())
    base["datasets"] = [base["datasets"][-1]]
    speaker_ids = {
        dataset["speaker"].removeprefix("vocalset_"): index
        for index, dataset in enumerate(yaml.safe_load((ROOT / "configs/diffsinger_score_native.yaml").read_text())["datasets"][:-1])
    }
    raw_root = WORK / "raw/vocalset_lexical"
    rows, datasets = [], []
    with ZipFile(INDEX_ARCHIVE) as archive:
        members = [
            info for info in archive.infolist()
            if (
                info.filename.startswith("data_by_singer/")
                and "/excerpts/" in info.filename
                and PurePosixPath(info.filename).name.endswith(
                    ("_row_straight.wav", "_row_vibrato.wav")
                )
            )
        ]
        grouped: dict[str, list[dict]] = {}
        for info in members:
            path = PurePosixPath(info.filename)
            speaker, technique = path.parts[1], path.parts[3]
            extracted = subprocess.run(
                ("unzip", "-p", str(ARCHIVE), info.filename),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
            if len(extracted) < 44:
                raise RuntimeError(f"failed to extract {info.filename}")
            audio, rate = sf.read(io.BytesIO(extracted), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            aligned, ctc_score = align_words(audio, rate, model, labels, device)
            speaker_raw = raw_root / speaker
            wavs = speaker_raw / "wavs"
            wavs.mkdir(parents=True, exist_ok=True)
            grouped.setdefault(speaker, [])
            for phrase_index, word_slice in enumerate((aligned[:8], aligned[8:])):
                start = max(0.0, word_slice[0]["start"] - 0.02)
                end = min(len(audio) / rate, word_slice[-1]["end"] + 0.03)
                first, last = round(start * rate), round(end * rate)
                identifier = f"vocalset_{speaker}_row_{technique}_{phrase_index}"
                target = wavs / f"{identifier}.wav"
                sf.write(target, audio[first:last], rate, subtype="PCM_24")
                duration = (last - first) / rate
                phones, durations = phrase_labels(word_slice, start, end)
                durations[-1] += duration - sum(durations)
                record = {
                    "id": identifier, "speaker": speaker, "technique": technique,
                    "text": " ".join(WORDS[:8] if phrase_index == 0 else WORDS[8:]),
                    "audio_path": str(target.relative_to(ROOT)), "duration_seconds": duration,
                    "ph_seq": phones, "ph_dur": durations,
                    "split": "validation" if speaker in VALIDATION_SPEAKERS else "train",
                    "license": "CC BY 4.0", "song_copyright": "public_domain",
                    "label_status": "known_lyrics_mms_ctc_inferred_timing",
                    "ctc_mean_log_score": ctc_score,
                    "training_role": "generic_lexical_score_native_singing_prior",
                }
                rows.append(record)
                grouped[speaker].append(record)

    for speaker, speaker_rows in grouped.items():
        raw = raw_root / speaker
        with (raw / "transcriptions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur"])
            writer.writeheader()
            for row in speaker_rows:
                writer.writerow({"name": row["id"], "ph_seq": " ".join(row["ph_seq"]), "ph_dur": " ".join(f"{x:.6f}" for x in row["ph_dur"])})
        datasets.append({
            "raw_data_dir": str(raw), "speaker": speaker, "spk_id": speaker_ids[speaker],
            "language": "gyu", "test_prefixes": [row["id"] for row in speaker_rows if row["split"] != "train"],
        })

    manifest = ROOT / "data/manifests/diffsinger_vocalset_lexical.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    old_dictionary = Path(base["dictionaries"]["gyu"])
    dictionary = WORK / "dictionary-gyu-vocalset-lexical.txt"
    phones = {phone for row in rows for phone in row["ph_seq"]}
    phones |= {
        line.split("\t", 1)[0]
        for line in old_dictionary.read_text().splitlines()
        if line and line.split("\t", 1)[0] not in {"a", "e", "i", "o", "u"}
    }
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP"})))
    source = Path(base["finetune_ckpt_path"])
    checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_vocalset_lexical_vocab.ckpt"
    remap = remap_vocabulary(source, old_dictionary, dictionary, checkpoint)
    base["datasets"] += datasets
    base["dictionaries"]["gyu"] = str(dictionary)
    base["binary_data_dir"] = str(WORK / "binary_vocalset_lexical")
    base["finetune_ckpt_path"] = str(checkpoint)
    base["max_updates"] = 300
    base["val_check_interval"] = 100
    base["optimizer_args"]["lr"] = 1e-5
    config = ROOT / "configs/diffsinger_vocalset_lexical.yaml"
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    report = {
        "status": "compatible_lexical_singing_prior_ready", "rows": len(rows),
        "speakers": len(grouped), "train_rows": sum(row["split"] == "train" for row in rows),
        "validation_rows": sum(row["split"] != "train" for row in rows),
        "duration_minutes": round(sum(row["duration_seconds"] for row in rows) / 60, 3),
        "license": "CC BY 4.0", "lyrics": "known public-domain Row Your Boat lyrics",
        "timing": "MMS CTC inferred", "remap": remap,
        "ctc_score_mean": round(float(np.mean([row["ctc_mean_log_score"] for row in rows])), 4),
        "ctc_score_min": round(float(np.min([row["ctc_mean_log_score"] for row in rows])), 4),
        "config": str(config.relative_to(ROOT)),
    }
    (ROOT / "artifacts/reports/diffsinger_vocalset_lexical.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
