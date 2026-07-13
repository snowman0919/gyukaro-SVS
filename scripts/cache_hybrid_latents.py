#!/usr/bin/env python3
"""Cache frozen MOSS acoustic latents. Cache is reproducible and ignored by Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/hybrid_all.jsonl")
    parser.add_argument("--codec", default="data/cache/moss-audio-tokenizer-nano")
    parser.add_argument("--output", default="data/cache/hybrid_latents")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.codec, trust_remote_code=True).to(device).eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line]
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            target = output / f"{row['id']}.pt"
            if target.exists():
                continue
            audio, rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
            if rate != 48000:
                raise ValueError(f"{row['audio_path']}: expected 48000 Hz")
            waveform = torch.from_numpy(audio.mean(axis=1)).to(device)[None].repeat(2, 1)
            encoded = model.batch_encode([waveform])
            hidden = encoded.encoder_hidden_states[0].transpose(0, 1).float().cpu()
            torch.save(hidden, target)
            print(f"{index}/{len(rows)} {row['id']} {tuple(hidden.shape)}")


if __name__ == "__main__":
    main()
