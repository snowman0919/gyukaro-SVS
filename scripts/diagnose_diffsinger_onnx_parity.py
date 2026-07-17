#!/usr/bin/env python3
"""Cross-decode identical DiffSinger mels through PyTorch and ONNX vocoders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
DIFFSINGER = ROOT / "data/cache/diffsinger"


def waveform_stats(wav: np.ndarray) -> dict[str, float | int]:
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    return {
        "samples": int(wav.size),
        "peak": float(np.max(np.abs(wav))),
        "rms": float(np.sqrt(np.mean(np.square(wav)))),
        "clipped_samples": int(np.count_nonzero(np.abs(wav) >= 0.999)),
    }


def durations_from_mel2ph(mel2ph: torch.Tensor, token_count: int) -> np.ndarray:
    values = mel2ph.detach().cpu().numpy().reshape(-1)
    return np.asarray([(values == index).sum() for index in range(1, token_count + 1)], dtype=np.int64)[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=Path, required=True)
    parser.add_argument("--acoustic-onnx", type=Path, required=True)
    parser.add_argument("--vocoder-onnx", type=Path, required=True)
    parser.add_argument("--experiment", default="gtsinger_ja_tenor")
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--depth", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    hparams["T_start_infer"] = 1.0 - args.depth
    infer = DiffSingerAcousticInfer(load_model=True, load_vocoder=True, ckpt_steps=args.checkpoint)
    params = json.loads(args.ds.read_text(encoding="utf-8"))
    if len(params) != 1:
        raise ValueError("Parity diagnostic currently requires exactly one DS segment.")
    batch = infer.preprocess_input(params[0])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch_mel = infer.forward_model(batch).detach().cpu().numpy().astype(np.float32)
    tokens = batch["tokens"].detach().cpu().numpy().astype(np.int64)
    durations = durations_from_mel2ph(batch["mel2ph"], tokens.shape[1])
    f0 = batch["f0"].detach().cpu().numpy().astype(np.float32)

    acoustic = ort.InferenceSession(str(args.acoustic_onnx.resolve()), providers=["CPUExecutionProvider"])
    acoustic_inputs: dict[str, np.ndarray] = {
        "tokens": tokens,
        "durations": durations,
        "f0": f0,
        "depth": np.asarray(args.depth, dtype=np.float32),
        "steps": np.asarray(args.steps, dtype=np.int64),
    }
    input_names = {item.name for item in acoustic.get_inputs()}
    if "gender" in input_names:
        acoustic_inputs["gender"] = np.zeros_like(f0)
    if "velocity" in input_names:
        acoustic_inputs["velocity"] = np.ones_like(f0)
    onnx_mel = acoustic.run(["mel"], acoustic_inputs)[0].astype(np.float32)

    vocoder = ort.InferenceSession(str(args.vocoder_onnx.resolve()), providers=["CPUExecutionProvider"])

    def torch_vocode(mel: np.ndarray) -> np.ndarray:
        mel_tensor = torch.from_numpy(mel).to(infer.device)
        f0_tensor = torch.from_numpy(f0).to(infer.device)
        return infer.run_vocoder(mel_tensor, f0=f0_tensor)[0].detach().cpu().numpy().reshape(-1)

    def onnx_vocode(mel: np.ndarray) -> np.ndarray:
        return vocoder.run(["waveform"], {"mel": mel, "f0": f0})[0].reshape(-1)

    combinations = {
        "torch_mel_torch_vocoder": torch_vocode(torch_mel),
        "torch_mel_onnx_vocoder": onnx_vocode(torch_mel),
        "onnx_mel_torch_vocoder": torch_vocode(onnx_mel),
        "onnx_mel_onnx_vocoder": onnx_vocode(onnx_mel),
    }
    sample_rate = int(hparams["audio_sample_rate"])
    for name, wav in combinations.items():
        sf.write(args.output / f"{name}.wav", wav, sample_rate, subtype="FLOAT")
    np.save(args.output / "torch_mel.npy", torch_mel)
    np.save(args.output / "onnx_mel.npy", onnx_mel)
    np.save(args.output / "f0.npy", f0)

    report = {
        "ds": str(args.ds.resolve()),
        "acoustic_onnx": str(args.acoustic_onnx.resolve()),
        "vocoder_onnx": str(args.vocoder_onnx.resolve()),
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "depth": args.depth,
        "steps": args.steps,
        "tokens": tokens.reshape(-1).tolist(),
        "durations": durations.reshape(-1).tolist(),
        "frames": int(f0.shape[1]),
        "mel": {
            "torch": {
                "min": float(torch_mel.min()),
                "max": float(torch_mel.max()),
                "mean": float(torch_mel.mean()),
                "std": float(torch_mel.std()),
            },
            "onnx": {
                "min": float(onnx_mel.min()),
                "max": float(onnx_mel.max()),
                "mean": float(onnx_mel.mean()),
                "std": float(onnx_mel.std()),
            },
            "mean_absolute_difference": float(np.mean(np.abs(torch_mel - onnx_mel))),
        },
        "waveforms": {name: waveform_stats(wav) for name, wav in combinations.items()},
    }
    (args.output / "parity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
