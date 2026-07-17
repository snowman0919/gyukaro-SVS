from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
from evaluate_ace_step_ja_source_probe import source_gate  # noqa: E402


def test_source_gate_requires_both_ja_phrases_and_no_repetition():
    assert source_gate([
        {"case": "quality_ja", "lyric_similarity": .93, "repetition": False},
        {"case": "heldout_ja", "lyric_similarity": .91, "repetition": False},
    ])
    assert not source_gate([
        {"case": "quality_ja", "lyric_similarity": .93, "repetition": False},
        {"case": "heldout_ja", "lyric_similarity": .89, "repetition": False},
    ])
    assert not source_gate([
        {"case": "quality_ja", "lyric_similarity": .93, "repetition": True},
        {"case": "heldout_ja", "lyric_similarity": .91, "repetition": False},
    ])
