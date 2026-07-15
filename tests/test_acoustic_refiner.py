import torch

from gyu_singer.model import VocalAcousticRefiner


def test_refiner_shape_and_bounded_residual():
    model = VocalAcousticRefiner(channels=16, blocks=2, adapter_rank=4)
    source = torch.randn(2, 4096).clamp(-1, 1)
    output = model(source, "gyu")
    assert output.shape == source.shape
    assert torch.max(torch.abs(output - source)) <= model.max_residual + 1e-6


def test_refiner_cannot_invent_silence():
    model = VocalAcousticRefiner(channels=16, blocks=2, adapter_rank=4)
    with torch.no_grad():
        model.output.bias.fill_(10)
    silence = torch.zeros(1, 4096)
    assert torch.equal(model(silence, "gyu"), silence)


def test_stage_freezing_is_separate():
    model = VocalAcousticRefiner(channels=16, blocks=2, adapter_rank=4)
    universal = model.train_stage("universal")
    assert universal > 0 and all(parameter.requires_grad == ("_adapter" not in name) for name, parameter in model.named_parameters())
    singing = model.train_stage("singing")
    assert 0 < singing < universal and all(parameter.requires_grad == ("singing_adapter" in name) for name, parameter in model.named_parameters())
    gyu = model.train_stage("gyu")
    assert gyu == singing and all(parameter.requires_grad == ("gyu_adapter" in name) for name, parameter in model.named_parameters())
