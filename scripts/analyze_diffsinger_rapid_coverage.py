#!/usr/bin/env python3
"""Measure whether GTSinger covers the independent ultra-rapid phrase."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_MORAS = 6
TARGET_REPEAT_SECONDS = 0.6081
TARGET_PHONES = (
    "i_ja", "k_ja", "i_ja", "ɡ_ja", "a_ja", "ts_ja", "ɨ_ja",
    "m_ja", "a_ja", "ɾ_ja", "ɯ_ja",
)
BOUNDARIES = {"SP", "AP"}


def normalized_phones(row: dict) -> list[str]:
    return ["SP" if phone == "<SP>" else "AP" if phone == "<AP>" else phone
            for phone in row["ph"]]


def selected_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["language"] == "Japanese"
            and row["singer"] == "JA-Soprano-1"
            and "#Control_Group#" in row["item_name"]]


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p05": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p05": round(float(np.quantile(array, .05)), 4),
        "median": round(float(np.quantile(array, .5)), 4),
        "p95": round(float(np.quantile(array, .95)), 4),
    }


def mora_windows(row: dict) -> list[float]:
    """Return durations of uninterrupted six-mora windows."""
    windows: list[float] = []
    run: list[float] = []
    for token, duration in zip(row["txt"], row["word_durs"]):
        if token in {"<SP>", "<AP>"}:
            run = []
            continue
        run.append(float(duration))
        if len(run) >= TARGET_MORAS:
            windows.append(sum(run[-TARGET_MORAS:]))
    return windows


def contiguous_sequences(phones: list[str], length: int) -> Counter[tuple[str, ...]]:
    result: Counter[tuple[str, ...]] = Counter()
    start = 0
    for index in range(len(phones) + 1):
        if index == len(phones) or phones[index] in BOUNDARIES:
            run = phones[start:index]
            result.update(tuple(run[offset:offset + length])
                          for offset in range(len(run) - length + 1))
            start = index + 1
    return result


def analyze(rows: list[dict]) -> dict:
    phone_durations: dict[str, list[float]] = {phone: [] for phone in set(TARGET_PHONES)}
    bigrams: Counter[tuple[str, ...]] = Counter()
    target_sequences = 0
    windows: list[float] = []
    qualifying_rows: list[dict] = []
    for row in rows:
        phones = normalized_phones(row)
        durations = [float(value) for value in row["ph_durs"]]
        for phone, duration in zip(phones, durations):
            if phone in phone_durations:
                phone_durations[phone].append(duration)
        bigrams.update(contiguous_sequences(phones, 2))
        target_sequences += contiguous_sequences(phones, len(TARGET_PHONES))[TARGET_PHONES]
        row_windows = mora_windows(row)
        windows.extend(row_windows)
        count = sum(value <= TARGET_REPEAT_SECONDS for value in row_windows)
        if count:
            qualifying_rows.append({"item_name": row["item_name"], "windows": count})

    target_bigrams = list(zip(TARGET_PHONES, TARGET_PHONES[1:]))
    fast = [row for row in rows if row["pace"] == "fast"]
    qualifying_count = sum(value <= TARGET_REPEAT_SECONDS for value in windows)
    return {
        "status": "coverage_measured",
        "corpus": {
            "rows": len(rows),
            "fast_rows": len(fast),
            "manual_phoneme_alignment": True,
        },
        "independent_target": {
            "moras_per_repeat": TARGET_MORAS,
            "repeat_seconds": TARGET_REPEAT_SECONDS,
            "moras_per_second": round(TARGET_MORAS / TARGET_REPEAT_SECONDS, 3),
            "phone_sequence": list(TARGET_PHONES),
        },
        "six_mora_window_duration_seconds": quantiles(windows),
        "windows_at_least_as_fast_as_target": qualifying_count,
        "rows_with_target_speed_windows": len(qualifying_rows),
        "target_speed_examples": qualifying_rows[:20],
        "exact_target_phone_sequence_occurrences": target_sequences,
        "target_bigram_occurrences": {
            f"{left} {right}": bigrams[(left, right)] for left, right in target_bigrams
        },
        "target_phone_duration_seconds": {
            phone: quantiles(values) for phone, values in sorted(phone_durations.items())
        },
        "interpretation": (
            "The score is outside observed six-mora timing coverage; targeted rapid "
            "coarticulation data is required."
            if qualifying_count == 0 else
            "Some timing coverage exists; inspect exact phone-transition coverage before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "data/external/raw/gtsinger-lfs/processed/Japanese/metadata.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/reports/diffsinger_rapid_coverage.json",
    )
    args = parser.parse_args()
    rows = selected_rows(json.loads(args.metadata.read_text()))
    report = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
