import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_diffsinger_gtsinger_ko_qualified import (
    corpus_summary,
    ctc_metrics,
    normalized_phones,
    normalized_text,
    row_rejections,
    song_splits,
)

GATES = json.loads((ROOT / "configs/gtsinger_ko_qualified_protocol.json").read_text())["row_gates"]


def source_row(name="Korean#KO-Soprano-2#Breathy#song-a#Control_Group#0000"):
    return {
        "item_name": name,
        "language": "Korean",
        "singer": "KO-Soprano-2",
        "pace": "fast",
        "range": "high",
        "txt": ["가", "나"],
        "ph": ["k_ko", "ɐ_ko", "n_ko", "ɐ_ko"],
        "ph_durs": [0.1, 0.9, 0.1, 0.9],
        "ep_pitches": [60, 60, 72, 72],
        "ep_notedurs": [1.0, 1.0, 1.0, 1.0],
        "wav_fn": "Korean/test.wav",
    }


def measured(**updates):
    value = {
        "audio_duration_seconds": 2.0,
        "clipping_samples": 0,
        "whisper_similarity": 0.9,
        "ctc_coverage": 0.95,
        "ctc_unknown_ratio": 0.0,
        "ctc_monotonic": True,
    }
    value.update(updates)
    return value


def test_source_row_passes_only_complete_multi_evidence_gate():
    assert row_rejections(source_row(), measured(), GATES) == []


@pytest.mark.parametrize(("updates", "reason"), [
    ({"audio_duration_seconds": 1.9}, "duration"),
    ({"clipping_samples": 1}, "clipping"),
    ({"whisper_similarity": 0.79}, "whisper"),
    ({"ctc_coverage": 0.89}, "ctc_coverage"),
    ({"ctc_unknown_ratio": 0.051}, "ctc_unknown"),
    ({"ctc_monotonic": False}, "ctc_monotonic"),
])
def test_source_row_rejects_each_failed_gate(updates, reason):
    assert reason in row_rejections(source_row(), measured(**updates), GATES)


def test_metadata_shape_and_duration_mismatch_are_rejected():
    row = source_row()
    row["ep_pitches"] = row["ep_pitches"][:-1]
    assert "metadata_shape" in row_rejections(row, measured(), GATES)
    assert "duration_alignment" in row_rejections(
        source_row(), measured(audio_duration_seconds=2.051), GATES
    )


def test_text_and_phone_normalization_are_stable():
    assert normalized_text(" 가, 나! ") == "가나"
    row = source_row()
    row["ph"] = ["<SP>", "k_ko", "ɐ_ko", "<AP>"]
    assert normalized_phones(row) == ["SP", "k_ko", "ɐ_ko", "AP"]


def test_corpus_minimum_is_fail_closed():
    row = source_row() | {
        "audio_duration_seconds": 10.0,
        "max_phone_duration_seconds": 1.2,
        "max_interval_semitones": 12.0,
    }
    minimums = {
        "rows": 2,
        "duration_seconds": 20.0,
        "fast_rows": 2,
        "high_register_rows": 2,
        "sustained_rows": 2,
        "large_interval_rows": 2,
    }
    assert corpus_summary([row], minimums)["status"] == "foundation_source_gate_reject"
    assert corpus_summary([row, row | {"item_name": row["item_name"] + "x"}], minimums)["training_allowed"] is True


def test_song_split_never_leaks_complete_song():
    rows = [source_row(f"Korean#KO-Soprano-2#Breathy#song-{index}#Control_Group#0000")
            for index in range(6)]
    splits = song_splits(rows)
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    by_song = {name.split("#")[3]: split for split, names in splits.items() for name in names}
    assert len(by_song) == 6


def test_ctc_metrics_require_complete_monotonic_target():
    log_probs = torch.log_softmax(
        torch.tensor([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0],
                       [8.0, 0.0, 0.0], [0.0, 0.0, 8.0]]]),
        -1,
    )
    result = ctc_metrics(log_probs, torch.tensor([[1, 2]]))
    assert result["ctc_monotonic"] is True
    assert result["ctc_unknown_ratio"] == 0.0
    assert 0.0 < result["ctc_coverage"] <= 1.0
