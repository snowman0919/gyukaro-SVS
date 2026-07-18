import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


SCRIPT = Path("scripts/prepare_truncated_identity_corpus.py")
TRAIN_SCRIPT = Path("scripts/train_truncated_identity_final_wav.py")


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
