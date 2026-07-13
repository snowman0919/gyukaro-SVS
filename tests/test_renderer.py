import numpy as np

from gyu_singer.renderer import Renderer, midi_hz


def test_midi_hz_reference_pitch():
    assert midi_hz(69) == 440.0


def test_loop_renderer_honors_score_duration():
    renderer = Renderer("checkpoints/gyu_v1_experimental.npz")
    audio = renderer.render(
        {
            "sample_rate": 48_000,
            "notes": [{"pitch": 60, "start": 0, "duration": 0.25, "dynamics": 0.8}],
        }
    )
    assert audio.dtype == np.float32
    assert len(audio) == int((0.25 + 0.08) * 48_000)
    assert np.max(np.abs(audio)) > 0
