from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
from probe_diffsinger_ja_content_source import build_source_row  # noqa: E402


def test_source_row_forces_lyrics_into_score_timing_and_unvoices_consonants():
    score = {
        "notes": [
            {"pitch": 60, "start": 0.0, "duration": 1.0, "lyric": "そら"},
            {"pitch": 67, "start": 1.0, "duration": 1.2, "lyric": "うた"},
        ]
    }
    phones = {"そら": ["s", "o", "r", "a"], "うた": ["u", "t", "a"]}

    row, evidence = build_source_row(score, lambda text: phones[text])

    durations = np.asarray(row["ph_dur"].split(), dtype=np.float64)
    f0 = np.asarray(row["f0_seq"].split(), dtype=np.float64)
    assert row["ph_seq"] == "s_ja o_ja ɾ_ja a_ja ɯ_ja t_ja a_ja"
    assert abs(durations.sum() - 2.2) < 1e-6
    assert len(f0) == round(2.2 / 0.02)
    assert np.any(f0 == 0)
    assert set(np.round(f0[f0 > 1], 2)) == {261.63, 392.0}
    assert evidence[0]["lyric"] == "そら"
    assert evidence[1]["start_seconds"] == 1.0


def test_source_row_rejects_score_gaps_in_bounded_probe():
    score = {
        "notes": [
            {"pitch": 60, "start": 0.0, "duration": 1.0, "lyric": "あ"},
            {"pitch": 60, "start": 1.2, "duration": 1.0, "lyric": "あ"},
        ]
    }

    try:
        build_source_row(score, lambda _: ["a"])
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("score gap must not be silently invented")
