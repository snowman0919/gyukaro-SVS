#!/usr/bin/env python3
"""Generate one whole lyric phrase with the locally cached ACE-Step teacher."""
from __future__ import annotations

import argparse
from pathlib import Path

from acestep.pipeline_ace_step import ACEStepPipeline


LANGUAGE = {"ko": "Korean", "en": "English", "ja": "Japanese"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=LANGUAGE, required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--style", default="neutral")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="data/cache/ace-step-checkpoint")
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    model = ACEStepPipeline(checkpoint_dir=args.checkpoint, dtype="bfloat16")
    model(audio_duration=args.duration, prompt=f"a cappella {LANGUAGE[args.language]} female singer, clean dry solo vocal, no instruments, {args.style}", lyrics=f"[verse]\n{args.lyrics}", infer_step=20, guidance_scale=7, manual_seeds=[101], save_path=str(output))


if __name__ == "__main__":
    main()
