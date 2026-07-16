#!/usr/bin/env python3
"""Translate the fixed rapid PJS probe into GTSinger's native Japanese IPA set."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/work/pjs/rapid_repeat_voiced.ds"
TARGET = ROOT / "data/external/work/gtsinger/rapid_repeat_voiced.ds"
PHONE_MAP = {
    "ja_i": "i_ja", "ja_k": "k_ja", "ja_g": "ɡ_ja", "ja_a": "a_ja",
    "ja_ts": "ts_ja", "ja_u": "ɯ_ja", "ja_m": "m_ja", "ja_r": "ɾ_ja",
}


def translate(sequence: str) -> str:
    return " ".join(PHONE_MAP[phone] for phone in sequence.split())


def main() -> None:
    rows = json.loads(SOURCE.read_text())
    for row in rows:
        row["ph_seq"] = translate(row["ph_seq"])
        row["spk_mix"] = {"gts_ja_soprano": 1.0}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
