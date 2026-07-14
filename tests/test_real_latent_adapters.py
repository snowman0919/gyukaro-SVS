import torch

from gyu_singer.inference.latent_adapter import SoulXRealLatentAdapters


def test_identity_and_style_use_separate_parameters():
    model = SoulXRealLatentAdapters()
    hidden = torch.randn(1, 4, 512)
    identity = torch.randn(1, 64)
    model(hidden, identity, None).sum().backward()
    assert all(parameter.grad is not None for parameter in model.identity.parameters())
    assert all(parameter.grad is None for parameter in model.style.parameters())
