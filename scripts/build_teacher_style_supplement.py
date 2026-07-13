#!/usr/bin/env python3
"""Build a small, separately tracked style supplement for teacher evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from build_teacher_benchmark import TEXTS, _segments


STYLE_REFERENCES = [
    ("dark", 220, "clean_natural_phrase"),
    ("emotional", 209, "sustained_expressive_phrase"),
]
TEXT_INDICES = (3, 9, 14)


def build_rows() -> list[dict]:
    segment_text = {row["source_index"]: row["text"] for row in _segments()}
    rows = []
    for language, texts in TEXTS.items():
        for style, source_index, reference_role in STYLE_REFERENCES:
            for number, text_index in enumerate(TEXT_INDICES, 1):
                rows.append({
                    "id": f"teacher_extra_{language}_{style}_{number:03d}",
                    "language": language,
                    "text": texts[text_index],
                    "style": style,
                    "style_prompt": style,
                    "reference_ids": [f"gyu_real_{source_index:06d}"],
                    "reference_role": reference_role,
                    "reference_audio_path": f"data/source/Korea Digital Media High School {source_index}.m4a",
                    "reference_text": segment_text[source_index],
                })
    return rows


def main() -> None:
    output = Path("configs/teachers/trilingual_style_supplement.jsonl")
    rows = build_rows()
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
