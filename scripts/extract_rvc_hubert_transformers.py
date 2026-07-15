#!/usr/bin/env python3
"""Extract RVC v2-compatible 768-d HuBERT features without fairseq runtime."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, HubertModel


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    experiment = Path(args.experiment)
    source = experiment / "1_16k_wavs"
    output = experiment / "3_feature768"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "data/cache/hubert-base-ls960"
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
    model = HubertModel.from_pretrained(checkpoint, torch_dtype=torch.float16).cuda().eval()
    for index, path in enumerate(sorted(source.glob("*.wav")), 1):
        target = output / f"{path.stem}.npy"
        if target.exists():
            continue
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 16000:
            raise RuntimeError(f"{path}: expected 16 kHz")
        inputs = extractor(audio.mean(1), sampling_rate=rate, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(inputs.input_values.cuda().half()).last_hidden_state
        value = hidden.squeeze(0).float().cpu().numpy()
        if not np.isfinite(value).all() or value.shape[1] != 768:
            raise RuntimeError(f"{path}: invalid feature shape {value.shape}")
        np.save(target, value, allow_pickle=False)
        if index % 25 == 0:
            print(f"features {index}", flush=True)
    print({"status": "complete", "rows": len(list(output.glob('*.npy')))})


if __name__ == "__main__":
    main()
