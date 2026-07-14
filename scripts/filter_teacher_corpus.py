#!/usr/bin/env python3
"""Keep only cross-teacher-agreed speech for weak representation distillation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def filter_rows(rows: list[dict], minimum_teachers: int = 3) -> list[dict]:
    teachers_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        teachers_by_id[row["id"]].add(row["teacher"])
    accepted = [
        row for row in rows
        if row.get("quality_status") == "teacher_gate_pass_unadmitted"
        and (row.get("teacher_agreement_score") or 0) >= 0.5
        and len(teachers_by_id[row["id"]]) >= minimum_teachers
    ]
    means: dict[tuple[str, str], float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in accepted:
        grouped[row["language"], row["teacher"]].append(float(row["overall_confidence"]))
    for key, scores in grouped.items():
        means[key] = sum(scores) / len(scores)
    primary = {
        language: max((teacher for lang, teacher in means if lang == language), key=lambda teacher: means[language, teacher])
        for language in {row["language"] for row in accepted}
    }
    output = []
    for row in accepted:
        confidence = float(row["overall_confidence"])
        agreement = float(row["teacher_agreement_score"])
        row = row | {
            "teacher_role": "primary_representation_teacher" if row["teacher"] == primary[row["language"]] else "supplementary_representation_teacher",
            "trust_weight": round(min(0.20, max(0.05, 0.05 + 0.10 * confidence + 0.05 * agreement)), 4),
            "training_use": "representation_distillation_only_not_singing_decoder",
        }
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--minimum-teachers", type=int, default=3)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    selected = filter_rows(rows, args.minimum_teachers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    print(f"selected={len(selected)} total={len(rows)} output={output}")


if __name__ == "__main__":
    main()
