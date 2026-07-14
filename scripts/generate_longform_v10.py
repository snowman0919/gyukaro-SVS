#!/usr/bin/env python3
"""Create reproducible two-minute KO/EN/JA OpenUtau release project."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


TEMPO, TICKS = 70, 480
KO = [
    "하늘을향해노래해", "작은빛을따라가자", "바람결에마음실어", "푸른별을바라보며",
    "오늘처럼함께걷자", "따뜻한꿈을그려봐", "새벽공기깨어나면", "우리목소리이어져",
    "멀리있는길끝까지", "반짝이는순간마다", "소중한맘기억할게", "다시노래피어나고", "끝없는빛향해가자",
]
EN = ["sing with me beneath the open sky", "carry our song across the quiet night"]
JA = ["空へ歌を届けよう", "光を追いかけよう"]
PITCH = ([0, 2, 4, 5, 4, 2, 0, 0], [0, 0, 3, 5, 7, 5, 2, 0], [0, -2, 0, 4, 2, 0, -1, 0])


def note(lyric: str, tone: int, position: int, vibrato: bool) -> dict:
    return {
        "position": position, "duration": TICKS, "tone": tone, "lyric": lyric,
        "pitch": {"data": [{"x": -40, "y": 0, "shape": "io"}, {"x": 40, "y": 0, "shape": "io"}], "snap_first": True},
        "vibrato": {"length": 45 if vibrato else 0, "period": 180, "depth": 35, "in": 10, "out": 10, "shift": 0, "drift": 0, "vol_link": 0},
    }


def main() -> None:
    rows = [("ko", text, 0, 60) for text in KO] + [("en", text.split(), 1, 62) for text in EN] + [("ja", text, 2, 64) for text in JA]
    parts, boundaries, position = [], [], 0
    for index, (language, text, track, base) in enumerate(rows):
        lyrics = list(text) if isinstance(text, str) else text
        if len(lyrics) < 8: lyrics += ["+" + lyrics[-1]] * (8 - len(lyrics))
        lyrics = lyrics[:8]
        pattern = PITCH[index % len(PITCH)]
        style = (0, 2, 3)[index % 3]
        parts.append({
            "duration": 8 * TICKS, "name": f"{language.upper()} phrase {index + 1:02d}", "track_no": track, "position": position,
            "notes": [note(lyric, base + pattern[note_index], note_index * TICKS, note_index == 5) for note_index, lyric in enumerate(lyrics)],
            "curves": [
                {"abbr": "pitd", "xs": [0, 4 * TICKS, 8 * TICKS], "ys": [0, 75 if index % 2 == 0 else -50, 0]},
                {"abbr": "gyus", "xs": [0, 8 * TICKS], "ys": [style, style]},
            ],
        })
        end = position + 8 * TICKS
        gap = 240 if index < len(rows) - 1 and index % 2 == 0 else 0
        if index < len(rows) - 1:
            boundaries.append({"tick": end, "seconds": end * 60 / TEMPO / TICKS, "expected_gap_seconds": gap * 60 / TEMPO / TICKS,
                               "continuous_pitch_expected": gap == 0 and rows[index + 1][3] == base})
        position = end + gap
    project = {
        "name": "GYU Singer v1.0 long-form release validation", "comment": "Two-minute native OpenUtau phrase-render/export test.", "ustx_version": "0.9",
        "expressions": {"gyus": {"name": "GYU style", "abbr": "gyus", "type": "Curve", "min": 0, "max": 5, "default_value": 0, "is_flag": False, "flag": "", "skip_output_if_default": False}},
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}], "tempos": [{"position": 0, "bpm": TEMPO}],
        "tracks": [{"singer": "GYU-SINGER", "phonemizer": "OpenUtau.Core.DefaultPhonemizer", "renderer_settings": {"renderer": "GYU-SINGER"}, "track_name": f"GYU {language}"} for language in ("KO", "EN", "JA")],
        "voice_parts": parts,
    }
    project_path = Path("examples/openutau_v10_longform.ustx")
    project_path.write_text(yaml.safe_dump(project, allow_unicode=True, sort_keys=False))
    report = {
        "project": str(project_path), "tempo": TEMPO, "notes": len(rows) * 8, "phrases": len(rows),
        "languages": {"ko": len(KO) * 8, "en": len(EN) * 8, "ja": len(JA) * 8},
        "duration_seconds": position * 60 / TEMPO / TICKS, "boundaries": boundaries,
        "contains_pitch_curve": True, "contains_vibrato": True, "styles": ["neutral", "breathy", "energetic"],
    }
    report_path = Path("artifacts/reports/longform_v10_manifest.json"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
