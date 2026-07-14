#!/usr/bin/env python3
"""Generate one whole lyric phrase with the locally cached ACE-Step teacher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acestep.pipeline_ace_step import ACEStepPipeline


LANGUAGE = {"ko": "Korean", "en": "English", "ja": "Japanese"}
RESULT = "__GYU_RESULT__"
ERROR = "__GYU_ERROR__"


def generate(model, language: str, lyrics: str, duration: float, style: str, output: str) -> None:
    if duration <= 0:
        raise ValueError("duration must be positive")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model(audio_duration=duration, prompt=f"a cappella {LANGUAGE[language]} female singer, clean dry solo vocal, no instruments, {style}", lyrics=f"[verse]\n{lyrics}", infer_step=20, guidance_scale=7, manual_seeds=[101], save_path=output)


def worker(model) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            generate(model, request["language"], request["lyrics"], float(request["duration"]), request["style"], request["output"])
            print(RESULT, json.dumps({"output": request["output"]}), flush=True)
        except Exception as error:
            print(ERROR, str(error), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=LANGUAGE)
    parser.add_argument("--lyrics")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--style", default="neutral")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", default="data/cache/ace-step-checkpoint")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    model = ACEStepPipeline(checkpoint_dir=args.checkpoint, dtype="bfloat16")
    if args.worker:
        worker(model); return
    if not all((args.language, args.lyrics, args.duration, args.output)):
        parser.error("--language, --lyrics, --duration, and --output are required outside --worker")
    generate(model, args.language, args.lyrics, args.duration, args.style, args.output)


if __name__ == "__main__":
    main()
