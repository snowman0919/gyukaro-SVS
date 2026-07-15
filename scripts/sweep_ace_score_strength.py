#!/usr/bin/env python3
"""Bounded ACE score-guide strength sweep for the failing large-interval case."""
from __future__ import annotations

import json
from pathlib import Path

from acestep.pipeline_ace_step import ACEStepPipeline


def main() -> None:
    root = Path("artifacts/reports/rc5_ace_score_strength"); root.mkdir(parents=True, exist_ok=True)
    score = json.loads(Path("examples/review_large_interval_ko.json").read_text()); guide = Path("artifacts/reports/rc5_ace_score_source/large_interval_ko/score_guide.wav"); duration = max(note["start"] + note["duration"] for note in score["notes"])
    model = ACEStepPipeline(checkpoint_dir="data/cache/ace-step-checkpoint", dtype="bfloat16"); rows = []
    for strength in (.15, .3, .45):
        output = root / f"large_interval_strength_{strength:g}.wav"
        model(audio_duration=duration, prompt="clean dry a cappella Korean female solo vocal, no instruments, follow reference melody and timing", lyrics="[verse]\n" + " ".join(note["lyric"] for note in score["notes"]), infer_step=20, guidance_scale=7, manual_seeds=[101], audio2audio_enable=True, ref_audio_strength=strength, ref_audio_input=str(guide), save_path=str(output))
        rows.append({"strength": strength, "path": str(output)})
    (root / "manifest.json").write_text(json.dumps({"status": "bounded_source_sweep", "case": "large_interval_ko", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__": main()
