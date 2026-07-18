import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from run_gtsinger_gyu_identity_diagnostic import (  # noqa: E402
    CASES,
    SEEDS,
    build_score_ds,
    distribution,
    gate_foundation,
    protocol_manifest,
    render_jobs,
    transcript_flags,
    validate_matrix,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(case: str, seed: int) -> dict:
    return {
        "case": case,
        "seed": seed,
        "audio_path": f"local/{case}_{seed}.wav",
        "audio_sha256": "a" * 64,
        "whisper_transcript": "가나다",
        "lyric_similarity": 1.0,
        "repetition_detected": False,
        "omission_detected": False,
        "pitch_mae_cents": 10.0,
        "pitch_p90_abs_cents": 20.0,
        "gross_error_over_600_cents": 0.0,
        "voicing_accuracy": 0.95,
        "clip_fraction": 0.0,
        "hf_spike_p99_over_median": 10.0,
        "sample_jump_p999": 0.05,
        "spectral_flux_p95": 0.1,
    }


def _protocol() -> dict:
    return {
        "cases": [{"id": case} for case in CASES],
        "seeds": list(SEEDS),
        "gates": {
            "lyric_similarity_min": 0.8,
            "pitch_p90_abs_cents_max": 100.0,
            "gross_error_over_600_cents_max": 0.05,
            "voicing_accuracy_min": 0.8,
            "clip_fraction_max": 0.0,
            "hf_spike_p99_over_median_max": 20.0,
            "sample_jump_p999_max": 0.11,
        },
    }


def test_score_ds_uses_one_inferred_phrase_timeline():
    score = {
        "language": "ko",
        "tempo": 120,
        "notes": [
            {"pitch": 60, "start": 0.0, "duration": 0.5, "lyric": "가"},
            {"pitch": 62, "start": 0.5, "duration": 0.5, "lyric": "나"},
        ],
    }

    row, metadata = build_score_ds(score)

    assert row["text"] == "가나"
    assert row["spk_mix"] == {"gts_ja_soprano": 1.0}
    assert row["f0_timestep"] == 0.02
    assert len(row["ph_seq"].split()) == len(row["ph_dur"].split())
    assert math.isclose(sum(map(float, row["ph_dur"].split())), 1.0, abs_tol=0.021)
    assert len(row["f0_seq"].split()) == 50
    assert metadata["timing_labels"] == "score_timed_inferred_split"
    assert metadata["target_f0_source"] == "nominal_score_f0_with_frontend_voicing"


def test_score_ds_inserts_silence_phone_for_phrase_gap():
    score = {
        "language": "ko",
        "tempo": 120,
        "notes": [
            {"pitch": 60, "start": 0.0, "duration": 0.5, "lyric": "가"},
            {"pitch": 62, "start": 1.0, "duration": 0.5, "lyric": "나"},
        ],
    }

    row, metadata = build_score_ds(score)
    phones = row["ph_seq"].split()
    durations = list(map(float, row["ph_dur"].split()))

    assert "SP" in phones
    assert durations[phones.index("SP")] == pytest.approx(0.5)
    assert sum(durations) == pytest.approx(1.5)
    assert metadata["silence_gap_frames"] == 25


def test_protocol_is_deterministic_and_freezes_five_cases_three_seeds():
    first = protocol_manifest(ROOT)
    second = protocol_manifest(ROOT)

    assert first == second
    assert [row["id"] for row in first["cases"]] == list(CASES)
    assert first["seeds"] == [7, 21, 42]
    assert len(first["identity_references"]) == 5
    assert all(len(row["score_sha256"]) == 64 for row in first["cases"])
    assert first["labels"] == "inferred score-timed phoneme split"
    assert first["production_runtime_modified"] is False


def test_protocol_restarts_on_unavailable_reported_diffsinger_revision():
    protocol = protocol_manifest(ROOT)

    assert protocol["protocol_revision"] == 2
    assert protocol["invalidated_protocol_revision"] == 1
    assert protocol["models"]["reported_diffsinger_revision"] == (
        "0619d61d5301c4340db442a15cf3e73e197e9101"
    )
    assert protocol["models"]["reported_revision_available"] is False
    assert protocol["models"]["diffsinger_revision"] == (
        "753b7cc622aadf802b3145d7bb8f7df4afa213c4"
    )
    assert protocol["protocol_restart_reason"] == "invalid_reported_diffsinger_revision"


def test_protocol_rejects_split_leakage():
    protocol = protocol_manifest(ROOT)
    protocol["adaptation_splits"]["validation_ids"].append(
        protocol["adaptation_splits"]["train_ids"][0]
    )

    with pytest.raises(ValueError, match="split leakage"):
        validate_protocol(protocol)


def test_distribution_reports_complete_statistics():
    assert distribution([1.0, 2.0, 3.0]) == {
        "mean": 2.0,
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "standard_deviation": pytest.approx(0.8164965809),
    }


def test_matrix_requires_every_case_seed_and_finite_metric():
    rows = [_row(case, seed) for case in CASES for seed in SEEDS]
    validate_matrix(rows, _protocol())

    with pytest.raises(ValueError, match="matrix mismatch"):
        validate_matrix(rows[:-1], _protocol())

    broken = copy.deepcopy(rows)
    broken[0]["pitch_mae_cents"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_matrix(broken, _protocol())


def test_matrix_rejects_missing_wav_when_file_check_is_enabled(tmp_path):
    rows = [_row(case, seed) for case in CASES for seed in SEEDS]

    with pytest.raises(ValueError, match="missing WAV"):
        validate_matrix(rows, _protocol(), root=tmp_path)


def test_transcript_flags_repetition_and_omission():
    expected = "빠르게노래하자아"

    assert transcript_flags(expected, expected) == {
        "repetition_detected": False,
        "omission_detected": False,
    }
    assert transcript_flags(expected, expected + expected)["repetition_detected"] is True
    assert transcript_flags(expected, "빠르게")["omission_detected"] is True
    assert transcript_flags(expected, "와우 와우 와우 와우")["repetition_detected"] is True
    assert transcript_flags("빛을따라", "다아... 다아...")["repetition_detected"] is True


def test_render_jobs_freeze_deterministic_case_seed_paths():
    jobs = render_jobs(_protocol())

    assert len(jobs) == 15
    assert len({(row["case"], row["seed"]) for row in jobs}) == 15
    assert jobs[0]["title"] == "quality_ko_foundation_seed7"
    assert jobs[-1]["audio_path"].endswith("phrase_boundary_ko_foundation_seed42.wav")


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("lyric_similarity", 0.79, "foundation_content_failure"),
        ("repetition_detected", True, "foundation_content_failure"),
        ("pitch_p90_abs_cents", 101.0, "pitch_regression"),
        ("voicing_accuracy", 0.79, "voicing_regression"),
        ("clip_fraction", 0.001, "clipping_failure"),
        ("hf_spike_p99_over_median", 20.1, "artifact_regression"),
        ("sample_jump_p999", 0.111, "artifact_regression"),
    ],
)
def test_one_failed_seed_rejects_entire_foundation(field, value, failure):
    rows = [_row(case, seed) for case in CASES for seed in SEEDS]
    rows[-1][field] = value

    result = gate_foundation(rows, _protocol())

    assert result["status"] == "foundation_ko_gate_reject"
    assert result["pass_count"] == 14
    assert result["pass_ratio"] == pytest.approx(14 / 15)
    assert failure in result["failures"][-1]["failure_taxonomy"]


def test_all_rows_must_pass_before_training_is_allowed():
    rows = [_row(case, seed) for case in CASES for seed in SEEDS]

    result = gate_foundation(rows, _protocol())

    assert result["status"] == "foundation_ko_gate_pass"
    assert result["pass_count"] == 15
    assert result["identity_training_allowed"] is True
