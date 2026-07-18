import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("scripts/probe_truncated_identity_grad.py")


def load_probe():
    assert SCRIPT.exists(), "truncated identity gradient probe is not implemented"
    spec = importlib.util.spec_from_file_location("truncated_identity_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_truncated_reverse_matches_full_and_only_adapter_gets_gradient():
    probe = load_probe()
    torch.manual_seed(4)
    flow = probe.ToyFlow().eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    adapter = torch.nn.Linear(3, 3, bias=False)
    prompt = torch.randn(1, 2, 3)
    target = torch.randn(1, 4, 3)
    frozen = torch.cat([prompt, adapter(target).detach()], 1)
    trainable = torch.cat([prompt, adapter(target)], 1)
    torch.manual_seed(9)
    expected = probe.full_reverse_diffusion(flow, prompt, frozen, 8, 2.0)
    torch.manual_seed(9)
    actual = probe.truncated_reverse_diffusion(flow, prompt, frozen, trainable, 8, 2, 2.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert torch.isfinite(adapter.weight.grad).all()
    assert adapter.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in flow.parameters())
