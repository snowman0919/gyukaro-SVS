#!/usr/bin/env python3
"""Measure 3-second held-vowel stability for the phrase-level hybrid renderer."""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gyu_singer.inference import HybridRenderer, load_hybrid_model
from gyu_singer.inference.codec import MossCodecDecoder

sys.path.insert(0, "data/cache/soulx-singer")
from preprocess.tools.f0_extraction import F0Extractor


VOWELS = {"ko": "아", "en": "ah", "ja": "あ"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gyu_hybrid_v0.2.pt")
    parser.add_argument("--report", default="artifacts/reports/hybrid_sustained_vowels.json")
    parser.add_argument("--sample-prefix", default="hybrid_sustain")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = HybridRenderer(load_hybrid_model(args.checkpoint, device), MossCodecDecoder("data/cache/moss-audio-tokenizer-nano", device), "data/processed/master/216.wav")
    f0 = F0Extractor("data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt", device=device, target_sr=24000, hop_size=480, verbose=False)
    output = {"duration_sec": 3.0, "f0_extractor": "SoulX RMVPE", "results": {}}
    Path("artifacts/samples").mkdir(exist_ok=True)
    for language, lyric in VOWELS.items():
        torch.manual_seed(31)
        path = Path(f"artifacts/samples/{args.sample_prefix}_{language}.wav")
        sf.write(path, renderer.render({"language": language, "tempo": 120, "sample_rate": 48000, "notes": [{"pitch": 60, "start": 0, "duration": 3, "lyric": lyric}]}), 48000)
        contour = f0.process(str(path), verbose=False)
        voiced = contour[contour > 1]
        output["results"][language] = {"path": str(path), "voiced_ratio": round(len(voiced) / len(contour), 4), "f0_cv": None if len(voiced) < 3 else round(float(np.std(voiced) / np.mean(voiced)), 4), "median_f0_hz": None if len(voiced) == 0 else round(float(np.median(voiced)), 2)}
    Path(args.report).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
