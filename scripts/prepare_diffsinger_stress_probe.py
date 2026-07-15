#!/usr/bin/env python3
"""Build direct score/F0 DiffSinger inputs for RC6's worst stress cases."""
from __future__ import annotations

import json
from pathlib import Path

from gyu_singer.alignment.phrase import build_phrase_frames
from gyu_singer.frontend import phonemize


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "rapid_ko": ROOT / "examples/review_rapid_ko.json",
    "large_interval_ko": ROOT / "examples/review_large_interval_ko.json",
}


def main() -> None:
    output = ROOT / "artifacts/reports/diffsinger_score_native_pilot"
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "inputs_ready", "score_native": True, "cases": {}}
    for name, score_path in CASES.items():
        score = json.loads(score_path.read_text())
        text = "".join(note["lyric"] for note in score["notes"])
        frames = build_phrase_frames(
            phonemize(score["language"], text), score["notes"], frame_hz=50
        )
        phones = sorted(frames.phoneme_durations, key=lambda row: row["start_frame"])
        row = {
            "offset": 0,
            "text": text,
            "ph_seq": " ".join(phone["symbol"] for phone in phones),
            "ph_dur": " ".join(
                f'{phone["duration_frames"] / 50:.6f}' for phone in phones
            ),
            "f0_seq": " ".join(f"{value:.3f}" for value in frames.f0_hz.tolist()),
            "f0_timestep": "0.02",
            "spk_mix": {"gyu": 1.0},
        }
        target = output / f"{name}.ds"
        target.write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
        report["cases"][name] = {
            "score": str(score_path.relative_to(ROOT)),
            "input": str(target.relative_to(ROOT)),
            "seconds": round(sum(phone["duration_frames"] for phone in phones) / 50, 3),
            "phonemes": len(phones),
            "voiced_ratio": round(float(frames.voiced.mean()), 4),
            "timing_labels": "score_timed_inferred_split",
        }
    (output / "input_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
