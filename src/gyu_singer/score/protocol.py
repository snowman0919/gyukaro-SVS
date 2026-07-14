from __future__ import annotations

from copy import deepcopy


_CURVES = {"pitch", "dynamics", "breathiness", "tension", "brightness", "vibrato"}


def _curve_points(points: object, tempo: float, name: str) -> list[dict]:
    if points in (None, []):
        return []
    if not isinstance(points, list):
        raise ValueError(f"curves.{name} must be a list")
    normalized = []
    for point in points:
        if isinstance(point, dict):
            value = float(point["value"])
            time = float(point["time"]) if "time" in point else float(point["beat"]) * 60.0 / tempo
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            time, value = float(point[0]) * 60.0 / tempo, float(point[1])
        else:
            raise ValueError(f"curves.{name} point must be {{beat,value}}, {{time,value}}, or [beat,value]")
        if time < 0:
            raise ValueError(f"curves.{name} time must be non-negative")
        normalized.append({"time": time, "value": value})
    return sorted(normalized, key=lambda point: point["time"])


def normalize_score(score: dict) -> dict:
    """Validate renderer protocol v2 and deterministically convert beats to seconds."""
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
    curves = result.pop("curves", result.pop("expressions", {}))
    if not isinstance(curves, dict):
        raise ValueError("score.curves must be an object")
    unknown = set(curves) - _CURVES
    if unknown:
        raise ValueError(f"unsupported curves: {sorted(unknown)}")
    result["curves"] = {name: _curve_points(curves.get(name, []), result["tempo"], name) for name in _CURVES}
    style = result.get("style", {})
    if not isinstance(style, dict) or style.get("preset", "neutral") not in {"neutral", "soft", "breathy", "energetic", "dark", "bright", "tense", "vibrato"}:
        raise ValueError("style.preset is unsupported")
    result["style"] = {"preset": style.get("preset", "neutral")}
    notes = result.get("notes")
    if not isinstance(notes, list) or not notes:
        raise ValueError("score.notes must be a non-empty list")
    previous_end = 0.0
    for index, note in enumerate(notes):
        for key in ("pitch", "lyric"):
            if key not in note:
                raise ValueError(f"notes[{index}].{key} is required")
        if "start" not in note and "start_beat" not in note:
            raise ValueError(f"notes[{index}] needs start or start_beat")
        if "duration" not in note and "duration_beats" not in note:
            raise ValueError(f"notes[{index}] needs duration or duration_beats")
        note["pitch"] = float(note["pitch"])
        note["start"] = float(note["start"]) if "start" in note else float(note["start_beat"]) * 60.0 / result["tempo"]
        note["duration"] = float(note["duration"]) if "duration" in note else float(note["duration_beats"]) * 60.0 / result["tempo"]
        if not 0 <= note["pitch"] <= 127 or note["start"] < 0 or note["duration"] <= 0:
            raise ValueError(f"notes[{index}] has invalid pitch/start/duration")
        if note["start"] + 1e-6 < previous_end:
            raise ValueError("notes must be time ordered and non-overlapping")
        previous_end = note["start"] + note["duration"]
        note["lyric"] = str(note["lyric"])
        note["id"] = str(note.get("id", f"n{index + 1}"))
        note["slur"] = bool(note.get("slur", False))
    return result
