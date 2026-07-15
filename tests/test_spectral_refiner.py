import torch

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
