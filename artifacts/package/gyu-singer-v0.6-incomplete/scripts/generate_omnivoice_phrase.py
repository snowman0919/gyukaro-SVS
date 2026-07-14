#!/usr/bin/env python3
"""Generate one duration-locked multilingual lyric phrase with OmniVoice."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig


LANGUAGE = {"ko": "Korean", "en": "English", "ja": "Japanese"}
SEED = {"ko": 21, "en": 101, "ja": 101}
RESULT = "__GYU_RESULT__"
ERROR = "__GYU_ERROR__"


def generate(model: OmniVoice, language: str, lyrics: str, duration: float, output: str) -> None:
    if duration <= 0:
        raise ValueError("duration must be positive")
    torch.manual_seed(SEED[language])
    audio = model.generate(
        text=lyrics,
        language=LANGUAGE[language],
        duration=duration,
        generation_config=OmniVoiceGenerationConfig(
            guidance_scale=3.0, postprocess_output=False, pad_duration=0, fade_duration=0
        ),
    )[0]
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, model.sampling_rate)


def worker(model: OmniVoice) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            generate(model, request["language"], request["lyrics"], float(request["duration"]), request["output"])
            print(RESULT, json.dumps({"output": request["output"]}), flush=True)
        except Exception as error:
            print(ERROR, str(error), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=LANGUAGE)
    parser.add_argument("--lyrics")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", default="data/cache/omnivoice-checkpoint")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    model = OmniVoice.from_pretrained(args.checkpoint, device_map="cuda:0", dtype=torch.float16).eval()
    if args.worker:
        worker(model); return
    if not all((args.language, args.lyrics, args.duration, args.output)):
        parser.error("--language, --lyrics, --duration, and --output are required outside --worker")
    generate(model, args.language, args.lyrics, args.duration, args.output)


if __name__ == "__main__":
    main()
