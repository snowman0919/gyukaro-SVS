import importlib.util
import inspect
from pathlib import Path

import numpy as np
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


def test_preservation_losses_are_zero_for_identical_audio():
    probe = load_probe()
    torch.manual_seed(12)
    audio = torch.randn(1, 12_000) * 0.01
    f0 = torch.full((25,), 220.0)
    losses = probe.preservation_losses(audio, audio, 48_000, f0)
    assert set(losses) == {
        "waveform", "stft_256", "stft_1024", "stft_4096", "pitch_period"
    }
    assert all(torch.isfinite(value) and value.abs() < 1e-7 for value in losses.values())


def test_gradient_audit_rejects_frozen_parameter_gradient():
    probe = load_probe()
    adapter = torch.nn.Linear(2, 2)
    frozen = torch.nn.Linear(2, 2)
    adapter(torch.ones(1, 2)).sum().backward()
    frozen.weight.grad = torch.ones_like(frozen.weight)
    audit = probe.gradient_audit(adapter, {"backbone": frozen})
    assert audit["adapter_gradient_finite"]
    assert audit["adapter_gradient_nonzero"]
    assert audit["unexpected_frozen_gradients"] == ["backbone.weight"]
    assert not audit["pass"]


def test_source_crop_places_waveform_and_f0_on_requested_device():
    probe = load_probe()
    assert "device" in inspect.signature(probe._source_crop).parameters

    class Extractor:
        def process(self, path, verbose=False):
            return np.ones(200, dtype=np.float32) * 220

    crop, f0, _ = probe._source_crop(
        "unused.wav", Extractor(), lambda path, rate: torch.zeros(1, rate * 5),
        24_000, 3.0, torch.device("cpu"),
    )
    assert crop.device == f0.device == torch.device("cpu")


def test_optional_peft_is_disabled_for_soulx_pinned_transformers():
    probe = load_probe()

    class WavlmModule:
        is_peft_available = staticmethod(lambda: True)

    module = WavlmModule()
    probe._disable_optional_peft(module)
    assert not module.is_peft_available()


def test_legacy_wavlm_weight_norm_keys_are_converted_without_loss():
    probe = load_probe()
    weight_g = torch.randn(1, 1, 3)
    weight_v = torch.randn(2, 4, 3)
    state = {
        "wavlm.encoder.pos_conv_embed.conv.weight_g": weight_g,
        "wavlm.encoder.pos_conv_embed.conv.weight_v": weight_v,
        "projector.weight": torch.randn(2, 2),
    }
    converted = probe._convert_legacy_wavlm_weight_norm(state)
    assert "wavlm.encoder.pos_conv_embed.conv.weight_g" not in converted
    assert "wavlm.encoder.pos_conv_embed.conv.weight_v" not in converted
    assert converted["wavlm.encoder.pos_conv_embed.conv.parametrizations.weight.original0"] is weight_g
    assert converted["wavlm.encoder.pos_conv_embed.conv.parametrizations.weight.original1"] is weight_v
