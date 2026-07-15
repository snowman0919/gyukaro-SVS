#!/usr/bin/env python3
"""Render independent Korean stress scores through the bounded GYU FiLM adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "data/cache/mlp-singer"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MODEL_ROOT))

from data.preprocess import Preprocessor  # noqa: E402
from train_mlp_singer_gyu_adapt import remap_model  # noqa: E402
from train_mlp_singer_gyu_film import GyuFilm, GyuResidual, hidden  # noqa: E402


@torch.no_grad()
def acoustic(model, adapter, notes, phonemes, strength: float) -> torch.Tensor:
    chunk = model.seq_len
    remainder = len(notes) % chunk
    if remainder:
        pad = chunk - remainder
        notes = torch.cat((notes, torch.zeros(pad, dtype=torch.long)))
        phonemes = torch.cat((phonemes, torch.zeros(pad, dtype=torch.long)))
    else:
        pad = 0
    notes = notes.reshape(-1, chunk).cuda()
    phonemes = phonemes.reshape(-1, chunk).cuda()
    h = hidden(model, notes, phonemes)
    adapted = h + strength * (adapter(h) - h)
    mel = model.proj(adapted).reshape(-1, model.proj.out_features)
    if pad:
        mel = mel[:-pad]
    return mel.transpose(0, 1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("mel", "speaker", "speaker_residual"), default="mel"
    )
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--strength", type=float, choices=(0.25, 0.5, 1.0), required=True)
    args = parser.parse_args()
    model, _ = remap_model()
    model.cuda().eval()
    sys.modules.pop("utils", None)
    sys.path.insert(0, str(MODEL_ROOT / "hifi-gan"))
    from env import AttrDict as HifiAttrDict
    from models import Generator

    allowed_steps = {
        "mel": {100, 400, 1000},
        "speaker": {25, 50, 100},
        "speaker_residual": {25, 50, 100},
    }
    if args.step not in allowed_steps[args.family]:
        parser.error(f"invalid step {args.step} for {args.family}")
    family_dir = {
        "mel": "gyu_film",
        "speaker": "gyu_speaker_film",
        "speaker_residual": "gyu_speaker_residual",
    }[args.family]
    checkpoint = torch.load(
        MODEL_ROOT / f"checkpoints/{family_dir}/steps_{args.step}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("adapter_type", "film") == "residual":
        adapter = GyuResidual(checkpoint["channels"], bottleneck=16, limit=checkpoint["limit"])
    else:
        adapter = GyuFilm(checkpoint["channels"], checkpoint["limit"])
    adapter = adapter.cuda().eval()
    adapter.load_state_dict(checkpoint["adapter"])
    preprocessor = Preprocessor(
        SimpleNamespace(
            **json.loads((ROOT / "configs/mlp_singer_preprocess_gyu.json").read_text())
        )
    )
    hifi_config = HifiAttrDict(
        json.loads((MODEL_ROOT / "hifi-gan/config.json").read_text())
    )
    vocoder = Generator(hifi_config).cuda().eval()
    vocoder.load_state_dict(
        torch.load(
            MODEL_ROOT / "hifi-gan/g_02500000", map_location="cpu", weights_only=False
        )["generator"]
    )
    vocoder.remove_weight_norm()
    prefix = {
        "mel": "film",
        "speaker": "speaker_film",
        "speaker_residual": "speaker_residual",
    }[args.family]
    label = f"{prefix}_{args.step}_s{round(args.strength * 100):03d}"
    output = ROOT / "artifacts/reports/mlp_singer_korean_probe/listening" / label
    output.mkdir(parents=True, exist_ok=True)
    for case in ("rapid_ko", "large_interval_ko"):
        data = MODEL_ROOT / "data/probe"
        notes, phonemes = preprocessor.prepare_inference(
            data / "mid" / f"{case}_gyu.mid", data / "txt" / f"{case}_gyu.txt"
        )
        mel = acoustic(model, adapter, notes, phonemes, args.strength)
        with torch.inference_mode():
            audio = vocoder(mel.cuda()).squeeze().cpu().numpy()
        target = output / f"{case}.wav"
        sf.write(target, audio, hifi_config.sampling_rate, subtype="PCM_16")
        print(target, flush=True)


if __name__ == "__main__":
    main()
