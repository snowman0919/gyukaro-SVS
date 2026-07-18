#!/usr/bin/env python3
"""Prepare a bounded CC0 Japanese speech prior for the PJS DiffSinger source."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from scipy.signal import resample_poly

from prepare_diffsinger_gyu_segments import remap_vocabulary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
CV = ROOT / "data/external/work/common_voice_ja17"
TRANSCRIPT = CV / "transcript/ja/train.tsv"
AUDIO = CV / "extracted/ja_train_0"
MAX_DURATION = 8.0
MIN_DURATION = 2.0


def resize_speaker_embedding(checkpoint: Path, speakers: int) -> dict:
    """Resize the DiffSinger speaker table without changing existing rows."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "model.fs2.spk_embed.weight"
    embedding = payload["state_dict"][key]
    old_speakers = embedding.shape[0]
    if speakers > old_speakers:
        added = embedding.mean(dim=0, keepdim=True).repeat(speakers - old_speakers, 1)
        resized = torch.cat((embedding, added), dim=0)
    else:
        resized = embedding[:speakers].clone()
    payload["state_dict"][key] = resized
    torch.save(payload, checkpoint)
    return {
        "old_speakers": old_speakers,
        "new_speakers": speakers,
        "preserved_row_max_abs_error": float((resized[:min(old_speakers, speakers)]
                                                - embedding[:min(old_speakers, speakers)]).abs().max()),
        "new_row_initialization": "mean:existing_speakers" if speakers > old_speakers else "not_applicable",
    }


def ctc_symbols(phone: str) -> str:
    """Map OpenJTalk phones to the MMS roman-character alignment alphabet."""
    return {"cl": "q"}.get(phone, phone).lower()


def phone_durations(
    phones: list[str], spans: list[tuple[int, int]], frames: int, duration: float
) -> tuple[list[str], list[float]]:
    """Merge character spans into phone durations while preserving clip length."""
    groups, cursor = [], 0
    for phone in phones:
        width = len(ctc_symbols(phone))
        group = spans[cursor:cursor + width]
        if len(group) != width:
            raise ValueError("incomplete CTC phone span")
        groups.append((group[0][0], group[-1][1]))
        cursor += width
    if cursor != len(spans) or not groups:
        raise ValueError("CTC span count does not match phonemes")
    seconds = duration / frames
    boundaries = [0.0]
    boundaries.extend((left[1] + right[0]) * seconds / 2 for left, right in zip(groups, groups[1:]))
    boundaries.append(duration)
    values = [after - before for before, after in zip(boundaries, boundaries[1:])]
    if min(values) <= 0:
        raise ValueError("non-positive phone duration")
    values[-1] += duration - sum(values)
    return [f"ja_{phone}" for phone in phones], values


def align(audio: np.ndarray, phones: list[str], model, labels: tuple[str, ...], device: str):
    target = "".join(ctc_symbols(phone) for phone in phones)
    dictionary = {label: index for index, label in enumerate(labels)}
    if set(target) - dictionary.keys():
        raise ValueError("phone is outside MMS alignment alphabet")
    with torch.inference_mode():
        emission, _ = model(torch.from_numpy(audio)[None].to(device))
    tokens = torch.tensor([[dictionary[char] for char in target]], device=device)
    alignment, scores = torchaudio.functional.forced_align(emission.log_softmax(-1), tokens)
    merged = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    if len(merged) != len(target):
        raise ValueError("incomplete CTC character alignment")
    spans = [(span.start, span.end) for span in merged]
    score = float(np.mean([span.score for span in merged]))
    symbols, durations = phone_durations(phones, spans, emission.shape[1], len(audio) / 16_000)
    return symbols, durations, score


def selected_rows(limit: int) -> list[dict]:
    rows = list(csv.DictReader(TRANSCRIPT.open(), delimiter="\t"))
    female = [row for row in rows if row["gender"] == "female_feminine"]
    speaker = Counter(row["client_id"] for row in female).most_common(1)[0][0]
    return [row for row in female if row["client_id"] == speaker]


def main() -> None:
    import pyopenjtalk

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=300)
    args = parser.parse_args()
    raw = WORK / "raw/common_voice_ja"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = torchaudio.pipelines.MMS_FA.get_labels()
    model = torchaudio.pipelines.MMS_FA.get_model().to(device).eval()
    accepted, rejected = [], Counter()
    for row in selected_rows(args.max_rows):
        if len(accepted) >= args.max_rows:
            break
        try:
            phones = pyopenjtalk.g2p(row["sentence"]).split()
            if not phones or "pau" in phones:
                rejected["internal_pause"] += 1
                continue
            source = AUDIO / row["path"]
            original, rate = librosa.load(source, sr=None, mono=True)
            duration = len(original) / rate
            if not MIN_DURATION <= duration <= MAX_DURATION:
                rejected["duration"] += 1
                continue
            analysis = resample_poly(original, 16_000, rate).astype("float32")
            symbols, durations, score = align(analysis, phones, model, labels, device)
            if score < -3.0:
                rejected["ctc_score"] += 1
                continue
            identifier = f"cvja_{len(accepted):04d}"
            target = wavs / f"{identifier}.wav"
            output = resample_poly(original, 44_100, rate).astype("float32")
            sf.write(target, output, 44_100, subtype="PCM_24")
            durations[-1] += len(output) / 44_100 - sum(durations)
            accepted.append({
                "id": identifier, "text": row["sentence"],
                "audio_path": str(target.relative_to(ROOT)),
                "duration_seconds": len(output) / 44_100,
                "ph_seq": symbols, "ph_dur": durations,
                "ctc_mean_log_score": score,
                "split": "validation" if len(accepted) % 10 == 0 else "train",
                "license": "CC0-1.0", "label_status": "OpenJTalk_G2P_MMS_CTC_inferred_timing",
                "training_role": "generic_Japanese_lexical_speech_prior",
            })
        except Exception as error:
            rejected[type(error).__name__] += 1

    if len(accepted) < args.max_rows:
        raise RuntimeError(f"only {len(accepted)} of {args.max_rows} Common Voice rows passed")
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "ph_seq", "ph_dur"))
        writer.writeheader()
        for row in accepted:
            writer.writerow({"name": row["id"], "ph_seq": " ".join(row["ph_seq"]),
                             "ph_dur": " ".join(f"{value:.7f}" for value in row["ph_dur"])})

    manifest = ROOT / "data/manifests/diffsinger_common_voice_ja.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted))
    base = yaml.safe_load((ROOT / "configs/diffsinger_pjs_compact_stress.yaml").read_text())
    old_dictionary = Path(base["dictionaries"]["gyu"])
    dictionary = WORK / "dictionary-pjs-common-voice.txt"
    old = {line.split("\t", 1)[0] for line in old_dictionary.read_text().splitlines() if line}
    phones = old | {phone for row in accepted for phone in row["ph_seq"] if phone not in {"AP", "SP"}}
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phones)))
    source = ROOT / "data/cache/diffsinger/checkpoints/pjs_compact_stress/model_ckpt_steps_3000.ckpt"
    checkpoint = ROOT / "data/cache/diffsinger/checkpoints/pjs_common_voice_vocab.ckpt"
    remap = remap_vocabulary(source, old_dictionary, dictionary, checkpoint)
    speaker_remap = resize_speaker_embedding(checkpoint, 2)
    base.update({
        "dictionaries": {"gyu": str(dictionary)},
        "datasets": [
            {"raw_data_dir": str(WORK / "raw/pjs"), "speaker": "pjs", "spk_id": 0,
             "language": "gyu", "test_prefixes": [f"pjs{index:03d}" for index in range(91, 101)]},
            {"raw_data_dir": str(raw), "speaker": "cv_ja", "spk_id": 1,
             "language": "gyu", "test_prefixes": [row["id"] for row in accepted if row["split"] == "validation"]},
        ],
        "num_spk": 2,
        "binary_data_dir": str(WORK / "binary_pjs_common_voice"),
        "finetune_ckpt_path": str(checkpoint),
        "finetune_strict_shapes": True,
        "max_updates": 1500, "val_check_interval": 300,
        "optimizer_args": {"lr": 5e-5},
    })
    config = ROOT / "configs/diffsinger_pjs_common_voice.yaml"
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    report = {
        "status": "bounded_lexical_prior_ready", "rows": len(accepted),
        "train_rows": sum(row["split"] == "train" for row in accepted),
        "validation_rows": sum(row["split"] == "validation" for row in accepted),
        "duration_minutes": round(sum(row["duration_seconds"] for row in accepted) / 60, 3),
        "speakers": 2, "speaker_ids": {"pjs": 0, "cv_ja": 1},
        "speaker_selection": "largest female Common Voice Japanese train contributor",
        "license": "CC0-1.0", "source": "Common Voice 17.0 Japanese train mirror",
        "timing": "OpenJTalk G2P plus MMS CTC inferred",
        "ctc_score_mean": round(float(np.mean([row["ctc_mean_log_score"] for row in accepted])), 4),
        "ctc_score_min": round(float(np.min([row["ctc_mean_log_score"] for row in accepted])), 4),
        "rejected": dict(rejected), "vocabulary_remap": remap,
        "speaker_embedding_remap": speaker_remap,
        "config": str(config.relative_to(ROOT)),
        "decision_rule": "run exact 1.8207-second rapid gate before expanding corpus or adapting GYU identity",
    }
    output = ROOT / "artifacts/reports/diffsinger_common_voice_ja.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
