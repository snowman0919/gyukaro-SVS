#!/usr/bin/env python3
"""Combine per-teacher manifests for cross-teacher evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def combine(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["teacher"], row["id"])
            if key in seen:
                raise ValueError(f"duplicate teacher item: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = combine([Path(path) for path in args.input])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"combined={len(rows)} output={output}")


if __name__ == "__main__":
    main()
