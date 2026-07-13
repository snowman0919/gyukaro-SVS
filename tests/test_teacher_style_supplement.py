import json
import subprocess
from collections import Counter
from pathlib import Path


def test_style_supplement_covers_missing_conditions():
    subprocess.run(["python", "scripts/build_teacher_style_supplement.py"], check=True)
    rows = [json.loads(line) for line in Path("configs/teachers/trilingual_style_supplement.jsonl").read_text().splitlines()]
    assert Counter(row["language"] for row in rows) == {"ko": 6, "en": 6, "ja": 6}
    assert {row["style_prompt"] for row in rows} == {"dark", "emotional"}
    assert {row["reference_role"] for row in rows} == {"clean_natural_phrase", "sustained_expressive_phrase"}
