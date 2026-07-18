import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path("scripts/prepare_truncated_identity_corpus.py")


def load_builder():
    assert SCRIPT.exists(), "truncated identity corpus builder is not implemented"
    spec = importlib.util.spec_from_file_location("truncated_identity_corpus", SCRIPT)
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
