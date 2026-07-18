import numpy as np
import torch

from gyu_singer.inference.spectral_gate import stationary_gate
from gyu_singer.inference.rc6 import GyuSingerRC6Renderer
from gyu_singer.inference.rc8 import GyuSingerRC8Renderer
from gyu_singer.model import SpectralAcousticRefiner


def test_spectral_refiner_is_identity_initialized():
    model = SpectralAcousticRefiner(channels=8, bottleneck_channels=32, blocks=2, adapter_rank=4)
    source = torch.randn(2, 4096) * 0.05
    with torch.inference_mode():
        output = model(source)
    assert output.shape == source.shape
    assert torch.max(torch.abs(output - source)).item() < 1e-5


def test_spectral_refiner_stage_freezing():
    model = SpectralAcousticRefiner(channels=8, bottleneck_channels=32, blocks=2, adapter_rank=4)
    universal = model.train_stage("universal")
    assert universal > 0
    assert all(
        parameter.requires_grad == ("_adapter" not in name)
        for name, parameter in model.named_parameters()
    )
    singing = model.train_stage("singing")
    assert 0 < singing < universal
    assert all(
        parameter.requires_grad == ("singing_adapter" in name)
        for name, parameter in model.named_parameters()
    )


def test_stationary_gate_preserves_boundaries_and_finds_sustained_vowel():
    sustained = {"language": "ko", "notes": [{"pitch": 64, "start": 0, "duration": 3, "lyric": "아"}]}
    gate = stationary_gate(sustained, 144_000)
    assert gate.mean() > .8 and gate[:2400].max() < .1
    interval = {"language": "ko", "notes": [{"pitch": 55, "start": 0, "duration": 1.2, "lyric": "높"}, {"pitch": 67, "start": 1.2, "duration": 1.2, "lyric": "이"}]}
    gate = stationary_gate(interval, 115_200)
    onset = round(1.2 * 48_000)
    assert np.max(gate[onset - 2400:onset + 2400]) < .1
    assert not gate.any()
    assert not stationary_gate({"language": "en", "notes": [{"pitch": 64, "start": 0, "duration": 3, "lyric": "ah"}]}, 144_000).any()


def test_rc8_uses_fixed_half_spectral_strength(monkeypatch):
    renderer = GyuSingerRC8Renderer.__new__(GyuSingerRC8Renderer)
    renderer.spectral_refiner = type("Refiner", (), {"process": lambda self, audio: audio + .2})()
    monkeypatch.setattr(GyuSingerRC6Renderer, "render", lambda self, score: np.zeros(1000, dtype="float32"))
    assert np.allclose(renderer.render({}), .1)


def test_rc8_uses_low_cfg_only_for_long_stable_notes():
    renderer = GyuSingerRC8Renderer.__new__(GyuSingerRC8Renderer)
    base = {"language": "ko", "style": {"preset": "neutral"}}
    sustained = base | {"notes": [{"pitch": 64, "start": 0, "duration": 5}]}
    rapid = base | {"notes": [{"pitch": 64, "start": 0, "duration": .25}]}
    interval = base | {"notes": [
        {"pitch": 55, "start": 0, "duration": 1.2},
        {"pitch": 67, "start": 1.2, "duration": 1.2},
    ]}

    assert renderer._decoder_options(sustained) == {"n_steps": 64, "cfg": 1.5, "seed": 21}
    assert renderer._decoder_options(rapid) == {"n_steps": 64, "cfg": 2.0, "seed": 21}
    assert renderer._decoder_options(interval) == {"n_steps": 50, "cfg": 2.0, "seed": 21}
