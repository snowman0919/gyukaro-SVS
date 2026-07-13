#!/usr/bin/env python3
"""Record candidate/accepted pseudo-singing manifests without promoting failed gates."""
from __future__ import annotations

import json
from pathlib import Path


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    root = Path("data/manifests")
    source = root / "pseudo_singing_pilot.jsonl"
    candidates = [json.loads(line) for line in source.read_text().splitlines() if line]
    accepted = [row for row in candidates if row.get("quality_status") == "accepted"]
    write(root / "pseudo_singing_candidates.jsonl", candidates)
    write(root / "pseudo_singing_accepted.jsonl", accepted)
    print({"candidates": len(candidates), "accepted": len(accepted)})


if __name__ == "__main__":
    main()
