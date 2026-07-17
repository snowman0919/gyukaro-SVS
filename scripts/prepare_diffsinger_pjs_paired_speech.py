#!/usr/bin/env python3
"""Add PJS same-singer paired speech as a lexical prior without identity mixing."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from scipy.signal import resample_poly

from prepare_diffsinger_common_voice_ja import align


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
PJS = ROOT / "data/external/work/pjs/PJS_corpus_ver1.1"


def source_phones(lines: list[str]) -> list[str]:
    rows = [line.split() for line in lines if line.strip()]
    return [row[2] for row in rows if row[2] not in {"pau", "xx"}]


def main() -> None:
    raw = WORK / "raw/pjs_speech"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    seconds = 0.0
    scores = []
    accepted = []
    rejected = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = torchaudio.pipelines.MMS_FA.get_labels()
    model = torchaudio.pipelines.MMS_FA.get_model().to(device).eval()
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "ph_seq", "ph_dur"))
        writer.writeheader()
        for index in range(1, 101):
            source_name = f"pjs{index:03d}"
            name = f"{source_name}_speech"
            source_dir = PJS / source_name
            source = source_dir / f"{source_name}_speech.wav"
            audio, rate = librosa.load(source, sr=None, mono=True)
            duration = len(audio) / rate
            try:
                phones, durations, score = align(
                    resample_poly(audio, 16_000, rate).astype(np.float32),
                    source_phones((source_dir / f"{source_name}.lab").read_text().splitlines()),
                    model, labels, device,
                )
            except ValueError as error:
                rejected.append({"id": name, "reason": str(error)})
                continue
            if score < -3.0:
                rejected.append({"id": name, "reason": "ctc_score_below_-3"})
                continue
            target = wavs / f"{name}.wav"
            if not target.exists():
                target.symlink_to(source)
            writer.writerow({"name": name, "ph_seq": " ".join(phones),
                             "ph_dur": " ".join(f"{value:.7f}" for value in durations)})
            seconds += duration
            scores.append(score)
            accepted.append(name)

    base = yaml.safe_load((ROOT / "configs/diffsinger_pjs_neutral_augmentation.yaml").read_text())
    base.update({
        "datasets": [
            {"raw_data_dir": str(WORK / "raw/pjs"), "speaker": "pjs", "spk_id": 0,
             "language": "gyu", "test_prefixes": [f"pjs{index:03d}" for index in range(91, 101)]},
            {"raw_data_dir": str(raw), "speaker": "pjs", "spk_id": 0,
             "language": "gyu", "test_prefixes": [name for name in accepted
                                                     if int(name[3:6]) >= 91]},
        ],
        "binary_data_dir": str(WORK / "binary_pjs_paired_speech"),
        "finetune_ckpt_path": str(
            ROOT / "data/cache/diffsinger/checkpoints/pjs_paired_speech/model_ckpt_steps_1500.ckpt"
        ),
        "finetune_strict_shapes": True,
        "max_updates": 2000,
        "val_check_interval": 500,
        "num_ckpt_keep": 4,
        "optimizer_args": {"lr": 5e-5},
    })
    config = ROOT / "configs/diffsinger_pjs_paired_speech.yaml"
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    report = {
        "status": "paired_same_singer_lexical_prior_ready",
        "singing_rows": 100,
        "speech_rows": len(accepted),
        "speech_rows_rejected": len(rejected),
        "rejected": rejected,
        "speech_duration_minutes": round(seconds / 60, 3),
        "speaker": "same PJS vocalist for singing and speech",
        "speech_alignment": "official PJS phone sequence plus MMS CTC inferred timing",
        "ctc_score_mean": round(float(np.mean(scores)), 4),
        "ctc_score_min": round(float(np.min(scores)), 4),
        "license": "CC BY-SA 4.0",
        "source_checkpoint": "paired singing/speech step 1500 before explicit unvoiced-F0 adaptation",
        "max_updates": 2000,
        "config": str(config.relative_to(ROOT)),
        "decision_rule": "exact rapid gate plus PJS091 heldout non-regression before final diffusion",
    }
    output = ROOT / "artifacts/reports/diffsinger_pjs_paired_speech.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
