#!/usr/bin/env python3
"""Export one OpenUtau USTX voice part to GYU renderer protocol v2, optionally render it."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import yaml


STYLE_PRESETS = ("neutral", "soft", "breathy", "energetic", "dark", "bright")


def _tick_to_seconds(tick: int, tempos: list[dict], resolution: int) -> float:
    ordered = sorted(tempos, key=lambda tempo: int(tempo.get("position", 0)))
    elapsed = 0.0
    position = 0
    bpm = float(ordered[0].get("bpm", 120.0))
    for tempo in ordered[1:]:
        change = int(tempo.get("position", 0))
        if tick <= change:
            break
        elapsed += (change - position) * 60.0 / bpm / resolution
        position, bpm = change, float(tempo.get("bpm", bpm))
    return elapsed + (tick - position) * 60.0 / bpm / resolution


def ustx_score(path: str | Path, language: str, part_index: int = 0) -> dict:
    project = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolution = int(project.get("resolution", 480))
    tempos = project.get("tempos") or [{"position": 0, "bpm": 120.0}]
    bpm = float(tempos[0].get("bpm", 120.0))
    parts = project.get("voice_parts") or []
    if not 0 <= part_index < len(parts):
        raise ValueError(f"voice part {part_index} does not exist")
    part = parts[part_index]
    origin = int(part.get("position", 0))
    notes = []
    for note in part.get("notes", []):
        lyric = str(note.get("lyric", "")).strip()
        if not lyric or lyric in {"R", "r", "rest", "-"}:
            continue
        start_tick = origin + int(note["position"])
        end_tick = start_tick + int(note["duration"])
        notes.append({"pitch": float(note["tone"]) + float(note.get("tuning", 0)) / 100,
                      "start": round(_tick_to_seconds(start_tick, tempos, resolution) - _tick_to_seconds(origin, tempos, resolution), 6),
                      "duration": round(_tick_to_seconds(end_tick, tempos, resolution) - _tick_to_seconds(start_tick, tempos, resolution), 6),
                      "lyric": lyric})
    if not notes:
        raise ValueError("selected voice part has no sung notes")
    curves: dict[str, list[dict]] = {name: [] for name in ("pitch", "dynamics", "breathiness", "tension")}
    style_values = []
    mapping = {"pitd": ("pitch", 0.01), "dyn": ("dynamics", 1.0), "brec": ("breathiness", 0.01), "tenc": ("tension", 0.01)}
    for curve in part.get("curves", []):
        abbr = curve.get("abbr")
        if abbr == "gyus":
            style_values.extend(float(value) for value in curve.get("ys", []))
            continue
        if abbr not in mapping:
            continue
        target, scale = mapping[abbr]
        curves[target].extend({"time": round(_tick_to_seconds(origin + int(x), tempos, resolution) - _tick_to_seconds(origin, tempos, resolution), 6),
                               "value": float(y) * scale}
                              for x, y in zip(curve.get("xs", []), curve.get("ys", [])))
    style_index = max(0, min(len(STYLE_PRESETS) - 1, round(sum(style_values) / len(style_values)))) if style_values else 0
    return {"protocol": "gyu-renderer-v2", "source": "OpenUtau USTX", "language": language, "tempo": bpm,
            "sample_rate": 48000, "notes": notes, "curves": curves, "style": {"preset": STYLE_PRESETS[style_index]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="OpenUtau .ustx YAML project")
    parser.add_argument("--language", choices=("ko", "en", "ja"), required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--output", required=True, help="renderer JSON output")
    parser.add_argument("--render-url", help="resident renderer /render endpoint")
    parser.add_argument("--wav", help="WAV output; requires --render-url")
    args = parser.parse_args()
    score = ustx_score(args.input, args.language, args.part)
    Path(args.output).write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n")
    if args.render_url:
        if not args.wav: parser.error("--wav is required with --render-url")
        request = urllib.request.Request(args.render_url.rstrip("/") + "/render", data=json.dumps(score).encode(), headers={"Content-Type": "application/json"}, method="POST")
        Path(args.wav).write_bytes(urllib.request.urlopen(request).read())


if __name__ == "__main__":
    main()
