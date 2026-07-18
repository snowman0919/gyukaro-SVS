#!/usr/bin/env python3
"""Qualify GTSinger Korean source rows before any DiffSinger training."""
from __future__ import annotations

import re


def normalized_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def normalized_phones(row: dict) -> list[str]:
    aliases = {"<SP>": "SP", "<AP>": "AP"}
    return [aliases.get(value, value) for value in row["ph"]]


def row_rejections(row: dict, measured: dict, gates: dict) -> list[str]:
    reasons: list[str] = []
    arrays = (
        row.get("ph"),
        row.get("ph_durs"),
        row.get("ep_pitches"),
        row.get("ep_notedurs"),
    )
    if (
        not all(isinstance(value, (list, tuple)) and value for value in arrays)
        or len({len(value) for value in arrays if value is not None}) != 1
    ):
        reasons.append("metadata_shape")

    duration = float(measured["audio_duration_seconds"])
    if not gates["duration_min_seconds"] <= duration <= gates["duration_max_seconds"]:
        reasons.append("duration")
    if (
        row.get("ph_durs")
        and abs(sum(map(float, row["ph_durs"])) - duration)
        > gates["duration_delta_max_seconds"]
    ):
        reasons.append("duration_alignment")
    if measured["clipping_samples"] > gates["clipping_samples_max"]:
        reasons.append("clipping")
    if measured["whisper_similarity"] < gates["whisper_similarity_min"]:
        reasons.append("whisper")
    if measured["ctc_coverage"] < gates["ctc_coverage_min"]:
        reasons.append("ctc_coverage")
    if measured["ctc_unknown_ratio"] > gates["ctc_unknown_ratio_max"]:
        reasons.append("ctc_unknown")
    if gates["ctc_monotonic_required"] and not measured["ctc_monotonic"]:
        reasons.append("ctc_monotonic")
    if not any(float(value) > 0 for value in row.get("ep_pitches", [])):
        reasons.append("no_voiced_note")
    if not any(value not in ("<SP>", "<AP>") for value in row.get("ph", [])):
        reasons.append("no_lexical_phone")
    return reasons


def corpus_summary(rows: list[dict], minimums: dict) -> dict:
    counts = {
        "rows": len(rows),
        "duration_seconds": sum(float(row["audio_duration_seconds"]) for row in rows),
        "fast_rows": sum(row["pace"] == "fast" for row in rows),
        "high_register_rows": sum(row["range"] == "high" for row in rows),
        "sustained_rows": sum(
            float(row["max_phone_duration_seconds"]) >= 1.0 for row in rows
        ),
        "large_interval_rows": sum(
            float(row["max_interval_semitones"]) >= 7.0 for row in rows
        ),
    }
    failures = [
        name for name, threshold in minimums.items() if counts[name] < threshold
    ]
    return {
        "status": (
            "source_qualification_pass"
            if not failures
            else "foundation_source_gate_reject"
        ),
        "counts": counts,
        "failed_minimums": failures,
        "training_allowed": not failures,
    }


def song_splits(rows: list[dict]) -> dict[str, list[str]]:
    songs = sorted({row["item_name"].split("#")[3] for row in rows})
    heldout = set(songs[-2:])
    validation = set(songs[-4:-2])
    split_by_song = {
        song: (
            "test" if song in heldout else "validation" if song in validation else "train"
        )
        for song in songs
    }
    result = {"train": [], "validation": [], "test": []}
    for row in rows:
        song = row["item_name"].split("#")[3]
        result[split_by_song[song]].append(row["item_name"])
    return {name: sorted(values) for name, values in result.items()}
