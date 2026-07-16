#!/usr/bin/env python3
"""Build the local-only RC9 full-song human listening gate."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/rc9_reference"
CASES = {12: "dense_rapid_a", 14: "dense_rapid_b", 33: "fixed_failure", 40: "rapid_refrain", 53: "ending_rapid"}
COMPARISONS = {40: "high_refrain", 47: "long_repetition", 53: "ending_refrain"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    source = WORK / "openutau_render.wav"
    project = yaml.safe_load((WORK / "nonbreath_oblige_gyu_rc9.ustx").read_text())
    requests = json.loads((WORK / "openutau_phrase_requests.json").read_text())
    output = WORK / "rc9_listening_gate"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    full = output / "01_full_openutau_rc9.wav"
    shutil.copy2(source, full)
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    bpm = float(project["tempos"][0]["bpm"])
    rows = [{"case": "full_song", "local_audio": str(full.relative_to(ROOT)), "sha256": sha256(full)}]
    for order, (index, label) in enumerate(CASES.items(), 2):
        start = float(project["voice_parts"][index - 1]["position"]) / 480 * 60 / bpm
        request = requests[index - 1]
        duration = max(float(note["start"]) + float(note["duration"]) for note in request["notes"])
        path = output / f"{order:02d}_{label}_phrase_{index:02d}.wav"
        sf.write(path, audio[round(start * rate):round((start + duration) * rate)], rate, subtype="PCM_24")
        rows.append({"case": label, "phrase": index, "local_audio": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    failed = WORK / "rc9_human_failed_baseline/openutau_render_failed.wav"
    if failed.exists():
        failed_audio, failed_rate = sf.read(failed, dtype="float32", always_2d=True)
        order = len(rows) + 1
        for index, label in COMPARISONS.items():
            start = float(project["voice_parts"][index - 1]["position"]) / 480 * 60 / bpm
            request = requests[index - 1]
            duration = max(float(note["start"]) + float(note["duration"]) for note in request["notes"])
            for variant, values, variant_rate in (("before", failed_audio, failed_rate), ("after", audio, rate)):
                path = output / f"{order:02d}_{label}_{variant}_phrase_{index:02d}.wav"
                sf.write(path, values[round(start * variant_rate):round((start + duration) * variant_rate)], variant_rate, subtype="PCM_24")
                rows.append({"case": f"{label}_{variant}", "phrase": index, "local_audio": str(path.relative_to(ROOT)), "sha256": sha256(path)})
                order += 1
    evaluation = json.loads((ROOT / "artifacts/reports/reference_song_rc9_evaluation.json").read_text())
    report = {
        "status": evaluation["status"],
        "backend": "gyu-singer-rc9",
        "required_checks": ["same melody and phrase timing", "rapid sections usable", "long repeated lyrics retained", "GYU identity retained", "no severe metallic or sustained artifact"],
        "files": rows,
        "copyright": "local evaluation derivatives; excluded from Git and package",
    }
    path = ROOT / "artifacts/reports/reference_song_rc9_listening_gate.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
