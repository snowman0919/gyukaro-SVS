import json
import subprocess
from collections import Counter
from pathlib import Path


def test_teacher_benchmark_has_fixed_trilingual_coverage():
    subprocess.run(["python", "scripts/build_teacher_benchmark.py"], check=True)
    rows = [json.loads(line) for line in Path("configs/teachers/trilingual_pilot.jsonl").read_text().splitlines()]
    assert Counter(row["language"] for row in rows) == {"ko": 100, "en": 100, "ja": 100}
    assert len({row["id"] for row in rows}) == 300
    assert len({row["reference_ids"][0] for row in rows}) == 5
