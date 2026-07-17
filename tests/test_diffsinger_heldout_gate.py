from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
from build_diffsinger_gtsinger_heldout_set import build_ds_row  # noqa: E402


def test_heldout_ds_row_preserves_manual_timing_and_rmvpe_grid():
    row = {"ph": ["i_ja", "<AP>"], "ph_durs": [.08, .02], "txt": ["い", "<AP>"]}
    result = build_ds_row(row, .10, np.array([261.63] * 5, dtype=np.float32))

    assert result["ph_seq"] == "i_ja AP"
    assert result["ph_dur"] == "0.0800000 0.0200000"
    assert result["text"] == "い"
    assert result["f0_timestep"] == 0.02
    assert len(result["f0_seq"].split()) == 5
