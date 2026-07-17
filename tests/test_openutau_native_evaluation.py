from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_openutau_diffsinger_native",
    ROOT / "scripts/evaluate_openutau_diffsinger_native.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_score_f0_zeros_unvoiced_and_uses_parent_tone():
    metrics = {"phoneme_timeline": [
        {"phoneme": "s_ja", "tone": 60, "start_ms": 0, "end_ms": 20},
        {"phoneme": "a_ja", "tone": 60, "start_ms": 20, "end_ms": 60},
        {"phoneme": "SP", "tone": 60, "start_ms": 60, "end_ms": 80},
    ]}
    target, present = MODULE.score_f0(metrics, 4)
    assert target[0] == 0
    assert np.allclose(target[1:3], 261.62555, atol=.01)
    assert target[3] == 0
    assert present.all()
