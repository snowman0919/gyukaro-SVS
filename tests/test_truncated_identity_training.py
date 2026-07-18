import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


SCRIPT = Path("scripts/prepare_truncated_identity_corpus.py")
TRAIN_SCRIPT = Path("scripts/train_truncated_identity_final_wav.py")
EVALUATE_SCRIPT = Path("scripts/evaluate_truncated_identity_candidates.py")


def load_builder():
    assert SCRIPT.exists(), "truncated identity corpus builder is not implemented"
    spec = importlib.util.spec_from_file_location("truncated_identity_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trainer():
    assert TRAIN_SCRIPT.exists(), "truncated final-WAV trainer is not implemented"
    spec = importlib.util.spec_from_file_location("truncated_identity_trainer", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluator():
    assert EVALUATE_SCRIPT.exists(), "truncated identity evaluator is not implemented"
    spec = importlib.util.spec_from_file_location("truncated_identity_evaluator", EVALUATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_split_excludes_collapsed_ja_and_protects_rapid_ko():
    split = load_builder().fixed_split()
    assert split == {
        "train": ["korean", "english", "japanese"],
        "validation": ["quality_ko", "quality_en", "quality_ja"],
        "heldout": [
            "heldout_ko", "heldout_en", "review_sustain_ko",
            "review_large_interval_ko", "review_phrase_boundary_ko",
        ],
        "protected": ["review_rapid_ko"],
        "excluded": ["heldout_ja"],
    }
    active = [set(values) for name, values in split.items() if name != "excluded"]
    assert all(not left & right for index, left in enumerate(active) for right in active[index + 1:])
    assert "heldout_ja" not in set().union(*active)
    assert "review_rapid_ko" not in set(split["train"] + split["validation"] + split["heldout"])


def test_capture_worker_copies_only_soulx_inputs(tmp_path):
    builder = load_builder()
    source = tmp_path / "source.wav"
    f0 = tmp_path / "f0.npy"
    identity = tmp_path / "identity.npy"
    style = tmp_path / "style.npy"
    warp = tmp_path / "warp.npy"
    source.write_bytes(b"wave")
    for path in (f0, identity, style, warp):
        np.save(path, np.ones(3, dtype=np.float32))
    worker = builder.CaptureWorker(tmp_path / "captured")
    with pytest.raises(builder.CaptureComplete):
        worker.request({
            "source": str(source), "f0_npy": str(f0), "identity_npy": str(identity),
            "style_npy": str(style), "content_warp_npy": str(warp),
            "content_warp_strength": 0.25, "n_steps": 32, "cfg": 1.5,
            "seed": 21, "output": str(tmp_path / "unused.wav"),
        })
    assert sorted(path.name for path in (tmp_path / "captured").iterdir()) == [
        "content_warp.npy", "f0.npy", "identity.npy", "source.wav", "style.npy",
    ]
    assert worker.options == {
        "content_warp_strength": 0.25, "n_steps": 32, "cfg": 1.5, "seed": 21,
    }


class TinyAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Linear(2, 2)
        self.style = nn.Linear(2, 2)


def test_optimizer_parameters_are_identity_only():
    trainer = load_trainer()
    adapters = TinyAdapters()
    selected = trainer.configure_identity_only(adapters)
    assert selected == list(adapters.identity.parameters())
    assert all(parameter.requires_grad for parameter in adapters.identity.parameters())
    assert all(not parameter.requires_grad for parameter in adapters.style.parameters())


def test_gradient_safety_rejects_frozen_nonfinite_and_zero_gradients():
    trainer = load_trainer()
    adapters = TinyAdapters()
    trainer.configure_identity_only(adapters)

    zero = trainer.gradient_safety(adapters.identity, {"style": adapters.style})
    assert zero["pass"] is False
    assert zero["reason"] == "zero_adapter_gradient"

    for parameter in adapters.identity.parameters():
        parameter.grad = torch.ones_like(parameter)
    next(adapters.identity.parameters()).grad.fill_(float("nan"))
    nonfinite = trainer.gradient_safety(adapters.identity, {"style": adapters.style})
    assert nonfinite["pass"] is False
    assert nonfinite["reason"] == "nonfinite_adapter_gradient"

    for parameter in adapters.identity.parameters():
        parameter.grad = torch.ones_like(parameter)
    next(adapters.style.parameters()).grad = torch.ones_like(next(adapters.style.parameters()))
    frozen = trainer.gradient_safety(adapters.identity, {"style": adapters.style})
    assert frozen["pass"] is False
    assert frozen["reason"] == "unexpected_frozen_gradient"


def test_update_safety_rejects_relative_update_and_total_drift():
    trainer = load_trainer()
    adapter = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        adapter.weight.fill_(1.0)
    initial = trainer.clone_parameters(adapter)
    before = trainer.clone_parameters(adapter)
    with torch.no_grad():
        adapter.weight.add_(0.01)
    too_large_step = trainer.update_safety(adapter, before, initial)
    assert too_large_step["pass"] is False
    assert too_large_step["reason"] == "relative_update_limit"

    with torch.no_grad():
        adapter.weight.fill_(1.0)
    before = trainer.clone_parameters(adapter)
    drifted_initial = {name: value - 0.06 for name, value in before.items()}
    too_much_drift = trainer.update_safety(adapter, before, drifted_initial)
    assert too_much_drift["pass"] is False
    assert too_much_drift["reason"] == "relative_drift_limit"


def _metric_row(phrase, seed, condition, wavlm, ecapa, protected=False):
    return {
        "phrase": phrase, "seed": seed, "condition": condition,
        "split": "protected" if protected else "heldout",
        "wavlm_similarity": wavlm, "ecapa_similarity": ecapa,
        "lyric_similarity": 0.98, "lyric_coverage": 0.98, "repeated_expected_span": None,
        "omission_detected": False, "pitch_mae_cents": 20.0,
        "voicing_accuracy": 0.96, "hf_spike_p99_over_median": 100.0,
        "sample_jump_p999": 0.02, "clipping_samples": 0,
    }


def test_summary_and_candidate_gate_require_all_identity_and_regression_thresholds():
    evaluator = load_evaluator()
    rows = []
    for phrase, protected in (("heldout", False), ("rapid", True)):
        for seed in (7, 21, 42):
            rows += [
                _metric_row(phrase, seed, "identity_off", 0.50, 0.60, protected),
                _metric_row(phrase, seed, "current_v07", 0.506, 0.606, protected),
                _metric_row(phrase, seed, "k2", 0.512, 0.612, protected),
            ]
    summary = evaluator.summarize([1.0, 2.0, 3.0])
    assert summary == {"mean": 2.0, "median": 2.0, "minimum": 1.0, "std": pytest.approx(0.8164965809)}
    result = evaluator.candidate_gates(rows, "k2")
    assert result["status"] == "human_pending"
    assert result["phrase_seed_pass_ratio"] == 1.0
    assert result["heldout_mean_delta_vs_identity_off"] == {
        "wavlm_similarity": pytest.approx(0.012),
        "ecapa_similarity": pytest.approx(0.012),
    }
    assert result["heldout_mean_delta_vs_current_v07"] == {
        "wavlm_similarity": pytest.approx(0.006),
        "ecapa_similarity": pytest.approx(0.006),
    }


def test_candidate_gate_rejects_one_phrase_seed_and_rapid_regression():
    evaluator = load_evaluator()
    rows = []
    for phrase, protected in (("heldout", False), ("rapid", True)):
        for seed in (7, 21, 42):
            rows += [
                _metric_row(phrase, seed, "identity_off", 0.50, 0.60, protected),
                _metric_row(phrase, seed, "current_v07", 0.506, 0.606, protected),
                _metric_row(phrase, seed, "k4", 0.512, 0.612, protected),
            ]
    bad = next(row for row in rows if row["condition"] == "k4" and row["phrase"] == "rapid" and row["seed"] == 21)
    bad["lyric_similarity"] = 0.90
    bad["hf_spike_p99_over_median"] = 120.0
    result = evaluator.candidate_gates(rows, "k4")
    assert result["status"] == "diagnostic_reject"
    assert result["phrase_seed_pass_ratio"] < 1.0
    assert result["rapid_ko_pass"] is False
    assert {failure["metric"] for failure in result["individual_failures"]} >= {
        "lyric_similarity", "hf_spike_p99_over_median",
    }


def test_rejected_checkpoint_is_deleted_but_human_pending_is_retained(tmp_path):
    evaluator = load_evaluator()
    rejected = tmp_path / "k2" / "identity_adapter_diagnostic.pt"
    retained = tmp_path / "k4" / "identity_adapter_diagnostic.pt"
    rejected.parent.mkdir(); retained.parent.mkdir()
    rejected.write_bytes(b"rejected"); retained.write_bytes(b"retained")
    disposition = evaluator.dispose_candidate_checkpoints(
        {"k2": rejected, "k4": retained},
        {"k2": {"status": "diagnostic_reject"}, "k4": {"status": "human_pending"}},
    )
    assert not rejected.exists()
    assert retained.exists()
    assert disposition["k2"]["deleted"] is True
    assert disposition["k4"]["deleted"] is False
