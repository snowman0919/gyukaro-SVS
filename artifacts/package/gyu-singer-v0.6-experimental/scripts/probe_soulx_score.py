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

sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")) / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
from soulxsinger.utils.audio_utils import load_wav
from soulxsinger.utils.file_utils import load_config
from gyu_singer.inference.latent_adapter import SoulXLatentAdapter


RESULT = "__GYU_RESULT__"
ERROR = "__GYU_ERROR__"


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


def initialize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("SoulX score probe requires CUDA")
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    model = SoulXSingerSVC(config).cuda()
    state = torch.load(args.model, weights_only=False, map_location="cpu")["state_dict"]
    model.load_state_dict(state); model.half(); model.mel.float(); model.eval()
    adapter = None
    if args.latent_adapter:
        saved_adapter = torch.load(args.latent_adapter, map_location="cuda", weights_only=False)
        adapter = SoulXLatentAdapter(**saved_adapter.get("config", {})).cuda().half().eval()
        adapter.load_state_dict(saved_adapter["model"])
        model._gyu_latent_adapter = adapter
        original_reverse = model.cfm_decoder.reverse_diffusion
        def adapted_reverse(pt_mel, pt_decoder_inp, gt_decoder_inp, n_timesteps=32, cfg=1):
            identity = getattr(model, "_gyu_identity", None)
            style = getattr(model, "_gyu_style", None)
            if identity is not None and style is not None:
                gt_decoder_inp = adapter(gt_decoder_inp, identity, style)
            return original_reverse(pt_mel, pt_decoder_inp, gt_decoder_inp, n_timesteps=n_timesteps, cfg=cfg)
        model.cfm_decoder.reverse_diffusion = adapted_reverse
    reference = load_wav(args.reference, config.audio.sample_rate).cuda()
    extractor = F0Extractor(args.rmvpe, device=device, target_sr=24000, hop_size=480, verbose=False)
    return model, config, reference, extractor.process(args.reference, verbose=False)


def render(model, config, reference, ref_f0, source_path: str, contour: np.ndarray, output: str, identity_path: str | None = None, style_path: str | None = None) -> None:
    source = load_wav(source_path, config.audio.sample_rate).cuda()
    expected = round(source.shape[-1] / 480)
    if len(contour) != expected:
        contour = np.interp(np.arange(expected), np.linspace(0, expected - 1, len(contour)), contour).astype(np.float32)
    if getattr(model, "_gyu_latent_adapter", None) is not None:
        model._gyu_identity = torch.from_numpy(np.load(identity_path).astype(np.float32)).cuda().half() if identity_path else None
        model._gyu_style = torch.from_numpy(np.load(style_path).astype(np.float32)).cuda().half() if style_path else None
    with torch.inference_mode():
        # Keep ablations causal: identical source/F0 uses identical diffusion noise.
        torch.manual_seed(21)
        audio, _ = model.infer(reference, source, torch.from_numpy(ref_f0).unsqueeze(0).cuda(), torch.from_numpy(contour).unsqueeze(0).cuda(), auto_shift=False, pitch_shift=0, n_steps=16, cfg=2.5, use_fp16=True)
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.squeeze().float().cpu().numpy(), config.audio.sample_rate)


def worker(model, config, reference, ref_f0) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            render(model, config, reference, ref_f0, request["source"], np.load(request["f0_npy"]).astype(np.float32), request["output"], request.get("identity_npy"), request.get("style_npy"))
            print(RESULT, json.dumps({"output": request["output"]}), flush=True)
        except Exception as error:
            print(ERROR, str(error), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--pitches", default="60,64,67,64")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--f0-npy", help="explicit 50 Hz score F0 contour; overrides --pitches")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    parser.add_argument("--model", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer/model-svc.pt")
    parser.add_argument("--config", default="data/cache/soulx-singer/soulxsinger/config/soulxsinger.yaml")
    parser.add_argument("--rmvpe", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--latent-adapter")
    args = parser.parse_args()
    model, config, reference, ref_f0 = initialize(args)
    if args.worker:
        worker(model, config, reference, ref_f0); return
    if not args.source or not args.output:
        parser.error("--source and --output are required outside --worker")
    source = load_wav(args.source, config.audio.sample_rate).cuda()
    if args.f0_npy:
        contour, notes = np.load(args.f0_npy).astype(np.float32), []
    else:
        contour, notes = score_f0(source.shape[-1] / config.audio.sample_rate, [int(x) for x in args.pitches.split(",")])
    render(model, config, reference, ref_f0, args.source, contour, args.output)
    if notes:
        Path(args.output).with_suffix(".score.json").write_text(json.dumps({"sample_rate": 24000, "notes": notes}, indent=2) + "\n")
    print(json.dumps({"output": args.output, "duration": round(source.shape[-1] / config.audio.sample_rate, 3), "notes": notes}))


if __name__ == "__main__":
    main()
