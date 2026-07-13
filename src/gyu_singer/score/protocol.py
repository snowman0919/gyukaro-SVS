from __future__ import annotations

from copy import deepcopy


def normalize_score(score: dict) -> dict:
    """Validate v2 score JSON. `start` and `duration` are seconds."""
    result = deepcopy(score)
    language = result.get("language")
    if language not in {"ko", "en", "ja"}:
        raise ValueError("score.language must be ko, en, or ja")
    result["sample_rate"] = int(result.get("sample_rate", 48000))
    if result["sample_rate"] != 48000:
        raise ValueError("hybrid SVS emits 48000 Hz")
    result["tempo"] = float(result.get("tempo", 120.0))
    if result["tempo"] <= 0:
        raise ValueError("score.tempo must be positive")
    notes = result.get("notes")
    if not isinstance(notes, list) or not notes:
        raise ValueError("score.notes must be a non-empty list")
    previous_end = 0.0
    for index, note in enumerate(notes):
        for key in ("pitch", "start", "duration", "lyric"):
            if key not in note:
                raise ValueError(f"notes[{index}].{key} is required")
        note["pitch"] = float(note["pitch"])
        note["start"] = float(note["start"])
        note["duration"] = float(note["duration"])
        if not 0 <= note["pitch"] <= 127 or note["start"] < 0 or note["duration"] <= 0:
            raise ValueError(f"notes[{index}] has invalid pitch/start/duration")
        if note["start"] + 1e-6 < previous_end:
            raise ValueError("notes must be time ordered and non-overlapping")
        previous_end = note["start"] + note["duration"]
        note["lyric"] = str(note["lyric"])
    result["expressions"] = result.get("expressions", {})
    return result
