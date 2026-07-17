from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
from evaluate_safe_duration_ja_probe import candidate_gates  # noqa: E402


def test_candidate_gates_require_lyrics_and_objective_nonregression():
    current = {
        "pitch_mae_cents": 7.32,
        "voicing_accuracy": 0.9189,
        "hf_spike_p99_over_median": 1493.5906,
        "sample_jump_p999": 0.099908,
        "wavlm_to_gyu": 0.80908,
        "ecapa_to_gyu": 0.2064,
    }
    candidate = {
        "asr_lyric_similarity": 1.0,
        "repetition_detected": False,
        "pitch_mae_cents": 7.99,
        "voicing_accuracy": 0.9371,
        "hf_spike_p99_over_median": 266.4469,
        "sample_jump_p999": 0.086318,
        "wavlm_to_gyu": 0.87407,
        "ecapa_to_gyu": 0.25973,
    }

    gates = candidate_gates(current, candidate)

    assert all(gates.values())
    assert not candidate_gates(current, candidate | {"asr_lyric_similarity": 0.899})[
        "heldout_lyric_similarity_at_least_090"
    ]
