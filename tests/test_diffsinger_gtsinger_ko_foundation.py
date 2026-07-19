import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_diffsinger_gtsinger_ko_foundation import (
    CASES,
    gate_row,
    select_checkpoint,
    summarize_checkpoint,
)


def protocol():
    return json.loads(
        (ROOT / "configs/gtsinger_ko_qualified_protocol.json").read_text()
    )


def passing_matrix(step=5000):
    return [
        {
            "case": case,
            "seed": seed,
            "checkpoint_step": step,
            "lyric_similarity": 0.95,
            "whisper_transcript": "정확한 노래",
            "repetition_detected": False,
            "omission_detected": False,
            "pitch_mae_cents": 10.0,
            "pitch_p90_abs_cents": 30.0,
            "gross_pitch_error_rate": 0.01,
            "voicing_accuracy": 0.95,
            "clipping_samples": 0,
            "hf_spike_ratio_to_source": 1.0,
            "sample_jump_ratio_to_source": 1.0,
            "waveform_discontinuity_ratio_to_source": 1.0,
            "stft_spike_ratio_to_source": 1.0,
            "audio_sha256": "a" * 64,
        }
        for case in CASES
        for seed in (7, 21, 42)
    ]


def test_complete_matrix_passes():
    result = summarize_checkpoint(passing_matrix(), protocol())
    assert result["status"] == "foundation_ko_gate_pass"
    assert result["pass_count"] == 21
    assert result["training_identity_allowed"] is True


def test_one_failed_seed_rejects_complete_checkpoint():
    rows = passing_matrix()
    rows[-1]["lyric_similarity"] = 0.899
    result = summarize_checkpoint(rows, protocol())
    assert result["status"] == "foundation_ko_gate_reject"
    assert result["pass_count"] == 20
    assert result["training_identity_allowed"] is False


@pytest.mark.parametrize(("field", "value", "failure"), [
    ("pitch_mae_cents", 35.01, "pitch_regression"),
    ("pitch_p90_abs_cents", 60.01, "pitch_regression"),
    ("gross_pitch_error_rate", 0.031, "pitch_regression"),
    ("voicing_accuracy", 0.899, "voicing_regression"),
    ("clipping_samples", 1, "clipping_failure"),
    ("hf_spike_ratio_to_source", 1.101, "artifact_regression"),
    ("sample_jump_ratio_to_source", 1.101, "artifact_regression"),
    ("waveform_discontinuity_ratio_to_source", 1.101, "artifact_regression"),
    ("stft_spike_ratio_to_source", 1.101, "artifact_regression"),
])
def test_each_preservation_failure_rejects_row(field, value, failure):
    row = passing_matrix()[0]
    row[field] = value
    assert failure in gate_row(row, protocol()["foundation_gates"])


@pytest.mark.parametrize("flag", ["repetition_detected", "omission_detected"])
def test_content_flags_reject_row(flag):
    row = passing_matrix()[0]
    row[flag] = True
    assert "foundation_content_failure" in gate_row(
        row, protocol()["foundation_gates"]
    )


def test_aggregate_pitch_mean_rejects_checkpoint():
    rows = passing_matrix()
    for row in rows:
        row["pitch_mae_cents"] = 20.01
    result = summarize_checkpoint(rows, protocol())
    assert result["aggregate_pitch_failure"] is True
    assert result["status"] == "foundation_ko_gate_reject"


def test_missing_case_or_seed_is_rejected():
    with pytest.raises(ValueError, match="matrix mismatch"):
        summarize_checkpoint(passing_matrix()[:-1], protocol())


@pytest.mark.parametrize(("field", "value"), [
    ("pitch_mae_cents", float("nan")),
    ("voicing_accuracy", float("inf")),
])
def test_non_finite_metric_is_rejected(field, value):
    rows = passing_matrix()
    rows[0][field] = value
    with pytest.raises(ValueError, match="non-finite"):
        summarize_checkpoint(rows, protocol())


def test_checkpoint_selection_is_lexicographic():
    reports = [summarize_checkpoint(passing_matrix(step), protocol()) for step in (5000, 10000, 15000)]
    reports[0]["minimum_lyric_similarity"] = 0.96
    reports[1]["minimum_lyric_similarity"] = 0.96
    reports[0]["maximum_pitch_mae_cents"] = 11.0
    reports[1]["maximum_pitch_mae_cents"] = 10.0
    reports[2]["minimum_lyric_similarity"] = 0.95
    assert select_checkpoint(reports)["selected_step"] == 10000
    reports[0]["maximum_pitch_mae_cents"] = 10.0
    assert select_checkpoint(reports)["selected_step"] == 5000


def test_selection_rejects_when_no_checkpoint_passes():
    report = summarize_checkpoint(passing_matrix(), protocol())
    report["status"] = "foundation_ko_gate_reject"
    assert select_checkpoint([report]) == {
        "status": "foundation_ko_gate_reject",
        "selected_step": None,
    }
