#!/usr/bin/env python3
"""Probe ACE-Step audio2audio as a score-timed phrase content source."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from acestep.pipeline_ace_step import ACEStepPipeline

LANGUAGE = {"ko": "Korean", "en": "English", "ja": "Japanese"}


def guide(score: dict, output: Path) -> float:
    rate = 44100; duration = max(note["start"] + note["duration"] for note in score["notes"]); audio = np.zeros(round(duration * rate), np.float32)
    for note in score["notes"]:
        start, end = round(note["start"] * rate), round((note["start"] + note["duration"]) * rate); time = np.arange(end - start) / rate
        frequency = 440 * 2 ** ((note["pitch"] - 69) / 12); tone = .1 * (np.sin(2 * np.pi * frequency * time) + .25 * np.sin(4 * np.pi * frequency * time))
        fade = min(round(.015 * rate), len(tone) // 2); tone[:fade] *= np.linspace(0, 1, fade); tone[-fade:] *= np.linspace(1, 0, fade); audio[start:end] += tone.astype(np.float32)
    sf.write(output, audio, rate, subtype="PCM_24"); return duration


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rc4", default="artifacts/reports/rc5_isolation"); parser.add_argument("--output", default="artifacts/reports/rc5_ace_score_source"); parser.add_argument("--checkpoint", default="data/cache/ace-step-checkpoint")
    args = parser.parse_args(); output = Path(args.output); shutil.rmtree(output, ignore_errors=True); output.mkdir(parents=True)
    matrix = json.loads((Path(args.rc4) / "matrix.json").read_text()); model = ACEStepPipeline(checkpoint_dir=args.checkpoint, dtype="bfloat16")
    report = {"status": "score_guide_source_probe_not_selected", "model": "ACE-Step-v1-3.5B audio2audio", "score_condition": "sine-harmonic note guide", "cases": {}}
    for case, data in matrix["cases"].items():
        score = json.loads(Path(data["score"]).read_text()); directory = output / case; directory.mkdir(); guide_path, target = directory / "score_guide.wav", directory / "ace_score_source.wav"; duration = guide(score, guide_path)
        model(audio_duration=duration, prompt=f"clean dry a cappella {LANGUAGE[score['language']]} female solo vocal, no instruments, follow reference melody and timing exactly", lyrics="[verse]\n" + " ".join(note["lyric"] for note in score["notes"]), infer_step=20, guidance_scale=7, manual_seeds=[101], audio2audio_enable=True, ref_audio_strength=.65, ref_audio_input=str(guide_path), save_path=str(target))
        report["cases"][case] = {"score": data["score"], "guide": str(guide_path), "source": str(target), "duration_seconds": duration}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({"output": str(output), "cases": len(report["cases"])}, indent=2))


if __name__ == "__main__": main()
