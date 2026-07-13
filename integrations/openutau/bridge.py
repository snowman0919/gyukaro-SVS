#!/usr/bin/env python3
"""Export one OpenUtau USTX voice part to GYU renderer protocol v2, optionally render it."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import yaml


def ustx_score(path: str | Path, language: str, part_index: int = 0) -> dict:
    project = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolution = int(project.get("resolution", 480))
    tempos = project.get("tempos") or [{"bpm": 120.0}]
    bpm = float(tempos[0].get("bpm", 120.0))
    parts = project.get("voice_parts") or []
    if not 0 <= part_index < len(parts):
        raise ValueError(f"voice part {part_index} does not exist")
    part = parts[part_index]
    origin = int(part.get("position", 0))
    seconds_per_tick = 60.0 / bpm / resolution
    notes = []
    for note in part.get("notes", []):
        lyric = str(note.get("lyric", "")).strip()
        if not lyric or lyric in {"R", "r", "rest", "-"}:
            continue
        notes.append({"pitch": float(note["tone"]), "start": round((origin + int(note["position"])) * seconds_per_tick, 6),
                      "duration": round(int(note["duration"]) * seconds_per_tick, 6), "lyric": lyric})
    if not notes:
        raise ValueError("selected voice part has no sung notes")
    return {"protocol": "gyu-renderer-v2", "source": "OpenUtau USTX", "language": language, "tempo": bpm, "sample_rate": 48000, "notes": notes}


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
