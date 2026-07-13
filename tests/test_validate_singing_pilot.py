import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("validate_singing_pilot", Path("scripts/validate_singing_pilot.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
contour_correlation = module.contour_correlation


def test_contour_correlation_handles_resampled_identical_pitch():
    target = np.array([120.0, 140.0, 160.0, 180.0])
    output = np.array([120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0])
    assert contour_correlation(target, output) > 0.99
