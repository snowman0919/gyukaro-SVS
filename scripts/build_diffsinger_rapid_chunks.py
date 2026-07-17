#!/usr/bin/env python3
"""Split the local rapid stress phrase at repeated-word boundaries."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/work/pjs/rapid_repeat_v3p5.ds"
OUT = ROOT / "data/external/work/pjs/rapid_chunks"
PHONES_PER_REPEAT = 11


def split_row(row: dict, repeats_per_chunk: int) -> list[dict]:
    phones, durations = row["ph_seq"].split(), row["ph_dur"].split()
    f0, velocity = row["f0_seq"].split(), row["velocity"].split()
    width = PHONES_PER_REPEAT * repeats_per_chunk
    total_duration = sum(map(float, durations))
    output = []
    for start in range(0, len(phones), width):
        end = min(start + width, len(phones))
        before = sum(map(float, durations[:start])) / total_duration
        through = sum(map(float, durations[:end])) / total_duration
        frame_start = round(before * len(f0))
        frame_end = round(through * len(f0))
        chunk = copy.deepcopy(row)
        chunk["offset"] = 0
        chunk["ph_seq"] = " ".join(phones[start:end])
        chunk["ph_dur"] = " ".join(durations[start:end])
        chunk["f0_seq"] = " ".join(f0[frame_start:frame_end])
        chunk["velocity"] = " ".join(velocity[frame_start:frame_end])
        output.append(chunk)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats-per-chunk", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    row = json.loads(SOURCE.read_text())[0]
    chunks = split_row(row, args.repeats_per_chunk)
    for index, chunk in enumerate(chunks):
        (OUT / f"r{args.repeats_per_chunk}_{index}.ds").write_text(
            json.dumps([chunk], ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps({"chunks": len(chunks), "repeats_per_chunk": args.repeats_per_chunk,
                      "phones": [len(row["ph_seq"].split()) for row in chunks],
                      "f0_frames": [len(row["f0_seq"].split()) for row in chunks]}))


if __name__ == "__main__":
    main()
