#!/usr/bin/env python3
"""Render phrase-level ACE vocals through SoulX SVC with an explicit score F0.

Run this only with ``.venv-soulx/bin/python``: SoulX pins an older Transformers
API than the project's normal runtime.  This is an evaluation probe, not a
dataset generator; it never modifies source recordings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")) / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
from soulxsinger.utils.audio_utils import load_wav
from soulxsinger.utils.file_utils import load_config


def hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def score_f0(duration: float, pitches: list[int]) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    frames = max(1, int(np.ceil(duration * 50)))
    boundaries = np.linspace(0, frames, len(pitches) + 1, dtype=int)
    contour = np.zeros(frames, dtype=np.float32)
    notes = []
    for start, end, pitch in zip(boundaries[:-1], boundaries[1:], pitches):
        contour[start:end] = hz(pitch)
        notes.append({"pitch": pitch, "start": round(start / 50, 3), "duration": round((end - start) / 50, 3)})
    return contour, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitches", default="60,64,67,64")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--f0-npy", help="explicit 50 Hz score F0 contour; overrides --pitches")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    parser.add_argument("--model", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer/model-svc.pt")
    parser.add_argument("--config", default="data/cache/soulx-singer/soulxsinger/config/soulxsinger.yaml")
    parser.add_argument("--rmvpe", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("SoulX score probe requires CUDA")
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    model = SoulXSingerSVC(config).cuda()
    state = torch.load(args.model, weights_only=False, map_location="cpu")["state_dict"]
    model.load_state_dict(state); model.half(); model.mel.float(); model.eval()
    reference = load_wav(args.reference, config.audio.sample_rate).cuda()
    source = load_wav(args.source, config.audio.sample_rate).cuda()
    extractor = F0Extractor(args.rmvpe, device=device, target_sr=24000, hop_size=480, verbose=False)
    ref_f0 = extractor.process(args.reference, verbose=False)
    if args.f0_npy:
        contour, notes = np.load(args.f0_npy).astype(np.float32), []
    else:
        contour, notes = score_f0(source.shape[-1] / config.audio.sample_rate, [int(x) for x in args.pitches.split(",")])
    if len(contour) != round(source.shape[-1] / 480):
        contour = np.resize(contour, round(source.shape[-1] / 480))
    with torch.inference_mode():
        audio, _ = model.infer(
            reference, source, torch.from_numpy(ref_f0).unsqueeze(0).cuda(),
            torch.from_numpy(contour).unsqueeze(0).cuda(),
            auto_shift=False, pitch_shift=0, n_steps=16, cfg=2.5, use_fp16=True,
        )
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio.squeeze().float().cpu().numpy(), config.audio.sample_rate)
    if notes:
        output.with_suffix(".score.json").write_text(json.dumps({"sample_rate": 24000, "notes": notes}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "duration": round(source.shape[-1] / config.audio.sample_rate, 3), "notes": notes}))


if __name__ == "__main__":
    main()
