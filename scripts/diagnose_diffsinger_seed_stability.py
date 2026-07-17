#!/usr/bin/env python3
"""Measure stochastic DiffSinger output stability without selecting a lucky sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
DIFFSINGER = ROOT / "data/cache/diffsinger"


def metrics(wav: np.ndarray, sample_rate: int) -> dict[str, float]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    spectrum = np.abs(librosa.stft(wav, n_fft=2048, hop_length=512)) + 1e-8
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    power = np.square(spectrum)
    hf = power[frequencies >= 8_000].sum(0) / np.maximum(power.sum(0), 1e-12)
    flatness = librosa.feature.spectral_flatness(S=spectrum).reshape(-1)
    return {
        "peak": float(np.max(np.abs(wav))),
        "rms": float(np.sqrt(np.mean(np.square(wav)))),
        "clip_fraction": float(np.mean(np.abs(wav) >= 0.999)),
        "hf_energy_ratio_mean": float(np.mean(hf)),
        "spectral_flatness_mean": float(np.mean(flatness)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=Path, required=True)
    parser.add_argument("--experiment", default="gtsinger_ja_tenor")
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.ds = args.ds.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DIFFSINGER))
    os.chdir(DIFFSINGER)
    from inference.ds_acoustic import DiffSingerAcousticInfer
    from utils.hparams import hparams, set_hparams

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], "--exp_name", args.experiment, "--infer"]
        set_hparams(print_hparams=False)
    finally:
        sys.argv = saved_argv
    infer = DiffSingerAcousticInfer(load_model=True, load_vocoder=True, ckpt_steps=args.checkpoint)
    params = json.loads(args.ds.read_text(encoding="utf-8"))
    if len(params) != 1:
        raise ValueError("Seed diagnostic currently requires one DS segment.")
    batch = infer.preprocess_input(params[0])
    rows = []
    sample_rate = int(hparams["audio_sample_rate"])
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mel = infer.forward_model(batch)
        wav = infer.run_vocoder(mel, f0=batch["f0"])[0].detach().cpu().numpy()
        path = args.output / f"seed_{seed:04d}.wav"
        sf.write(path, wav, sample_rate, subtype="FLOAT")
        rows.append({
            "seed": seed,
            "audio": str(path.resolve()),
            "mel_min": float(mel.min()),
            "mel_max": float(mel.max()),
            "mel_mean": float(mel.mean()),
            "mel_std": float(mel.std()),
        } | metrics(wav, sample_rate))
    summary = {
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "ds": str(args.ds.resolve()),
        "seed_count": args.seeds,
        "selection_policy": "none; all seeds retained",
        "rows": rows,
    }
    (args.output / "seed_stability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
