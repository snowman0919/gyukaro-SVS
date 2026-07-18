#!/usr/bin/env python3
"""Bounded final-WAV gradient feasibility probe for the v0.7 identity adapter."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F


FFT_SIZES = (256, 1024, 4096)


def diffusion_step(flow, state, prompt, condition, index: int, total_steps: int, cfg: float):
    prompt_len = prompt.shape[1]
    target_len = state.shape[1]
    step = 1.0 / total_steps
    mask = torch.ones(condition.shape[0], condition.shape[1], device=condition.device)
    target_mask = torch.ones(condition.shape[0], target_len, device=condition.device)
    time = ((index + 0.5) * step) * torch.ones(
        state.shape[0], dtype=state.dtype, device=state.device
    )
    predicted = flow.diff_estimator(torch.cat([prompt, state], 1), time, condition, mask)
    predicted = predicted[:, prompt_len:, :]
    if cfg > 0:
        unconditioned = flow.diff_estimator(
            state, time, torch.zeros_like(condition)[:, :target_len, :], target_mask
        )
        predicted_std = predicted.std()
        guided = predicted + cfg * (predicted - unconditioned)
        rescaled = guided * predicted_std / guided.std()
        predicted = 0.75 * rescaled + 0.25 * guided
    return state + predicted * step


def full_reverse_diffusion(flow, prompt, condition, total_steps: int, cfg: float):
    target_len = condition.shape[1] - prompt.shape[1]
    state = torch.randn(
        condition.shape[0], target_len, flow.mel_dim,
        dtype=condition.dtype, device=condition.device,
    )
    with torch.no_grad():
        for index in range(total_steps):
            state = diffusion_step(flow, state, prompt, condition, index, total_steps, cfg)
    return state


def truncated_reverse_diffusion(
    flow, prompt, cond_frozen, cond_trainable, total_steps: int, grad_steps: int, cfg: float
):
    if not 1 <= grad_steps <= total_steps:
        raise ValueError("grad_steps must be within total_steps")
    target_len = cond_frozen.shape[1] - prompt.shape[1]
    state = torch.randn(
        cond_frozen.shape[0], target_len, flow.mel_dim,
        dtype=cond_frozen.dtype, device=cond_frozen.device,
    )
    with torch.no_grad():
        for index in range(total_steps - grad_steps):
            state = diffusion_step(flow, state, prompt, cond_frozen, index, total_steps, cfg)
    state = state.detach()
    for index in range(total_steps - grad_steps, total_steps):
        state = diffusion_step(flow, state, prompt, cond_trainable, index, total_steps, cfg)
    return state


def _periodicity(audio, sample_rate: int, f0):
    hop = max(1, audio.shape[-1] // max(1, len(f0)))
    values = []
    for index, frequency in enumerate(f0):
        if frequency <= 0:
            continue
        period = int(round(sample_rate / float(frequency)))
        start = index * hop
        frame = audio[..., start:start + max(1024, period * 3)]
        if frame.shape[-1] <= period:
            continue
        left = frame[..., :-period] - frame[..., :-period].mean(-1, keepdim=True)
        right = frame[..., period:] - frame[..., period:].mean(-1, keepdim=True)
        values.append((left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8))
    return torch.stack(values).mean() if values else audio.sum() * 0


def preservation_losses(candidate, baseline, sample_rate: int, baseline_f0):
    losses = {"waveform": F.l1_loss(candidate, baseline)}
    for size in FFT_SIZES:
        window = torch.hann_window(size, device=candidate.device, dtype=candidate.dtype)
        candidate_stft = torch.stft(candidate, size, size // 4, window=window, return_complex=True)
        baseline_stft = torch.stft(baseline, size, size // 4, window=window, return_complex=True)
        losses[f"stft_{size}"] = F.l1_loss(
            torch.log(candidate_stft.abs().clamp_min(1e-7)),
            torch.log(baseline_stft.abs().clamp_min(1e-7)),
        )
    losses["pitch_period"] = (
        _periodicity(candidate, sample_rate, baseline_f0)
        - _periodicity(baseline, sample_rate, baseline_f0)
    ).abs()
    return losses


def gradient_audit(adapter, frozen_modules):
    adapter_gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
    finite = bool(adapter_gradients) and all(torch.isfinite(gradient).all() for gradient in adapter_gradients)
    norm = torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in adapter_gradients)) if adapter_gradients else torch.tensor(0.0)
    unexpected = [
        f"{module_name}.{parameter_name}"
        for module_name, module in frozen_modules.items()
        for parameter_name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    result = {
        "adapter_gradient_norm": float(norm),
        "adapter_gradient_finite": bool(finite),
        "adapter_gradient_nonzero": bool(norm > 0),
        "unexpected_frozen_gradients": unexpected,
    }
    result["pass"] = result["adapter_gradient_finite"] and result["adapter_gradient_nonzero"] and not unexpected
    return result


class ToyFlow(nn.Module):
    mel_dim = 3

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def diff_estimator(self, state, time, condition, mask):
        del mask
        return self.projection(state) + 0.1 * condition + time[:, None, None]


def _freeze(module):
    if isinstance(module, nn.Module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def _disable_optional_peft(wavlm_module):
    # ponytail: SoulX pins Transformers before EncoderDecoderCache; remove when its PEFT versions align.
    wavlm_module.is_peft_available = lambda: False


def _convert_legacy_wavlm_weight_norm(state):
    converted = dict(state)
    prefix = "wavlm.encoder.pos_conv_embed.conv."
    converted[prefix + "parametrizations.weight.original0"] = converted.pop(prefix + "weight_g")
    converted[prefix + "parametrizations.weight.original1"] = converted.pop(prefix + "weight_v")
    return converted


def _decode(model, prompt_mel, prompt_condition, target_condition, target_samples, steps, cfg):
    mel = model.cfm_decoder.reverse_diffusion(
        prompt_mel, prompt_condition, target_condition, n_timesteps=steps, cfg=cfg
    )
    audio = model.vocoder(mel.transpose(1, 2)[0:1]).squeeze().float()
    if audio.shape[-1] > target_samples:
        return audio[:target_samples]
    return F.pad(audio, (0, target_samples - audio.shape[-1]))


def _features(model, reference, source, reference_f0, source_f0):
    prompt_mel = model.mel(reference.float())
    prompt_pitch = model.f0_to_coarse(reference_f0)
    target_pitch = model.f0_to_coarse(source_f0)
    prompt_pitch = F.pad(prompt_pitch, (0, max(0, prompt_mel.shape[1] - prompt_pitch.shape[1])))
    prompt_pitch = prompt_pitch[:, :prompt_mel.shape[1]]
    pitch = torch.cat([prompt_pitch, target_pitch], 1)
    prompt_content = model.whisper_encoder.encode(reference, sr=model.audio_cfg.sample_rate)
    target_content = model.whisper_encoder.encode(source, sr=model.audio_cfg.sample_rate)
    prompt_content = F.pad(prompt_content, (0, 0, 0, max(0, prompt_pitch.shape[1] - prompt_content.shape[1])))
    target_content = F.pad(target_content, (0, 0, 0, max(0, target_pitch.shape[1] - target_content.shape[1])))
    content = torch.cat([
        prompt_content[:, :prompt_pitch.shape[1]], target_content[:, :target_pitch.shape[1]]
    ], 1)
    encoded = content + model.f0_encoder(pitch)
    return prompt_mel, encoded[:, :prompt_mel.shape[1]], encoded[:, prompt_mel.shape[1]:]


def _normalize_embedding(value):
    return F.normalize(value.flatten(1), dim=-1)


def _wavlm_inputs(audio_16k):
    return (audio_16k - audio_16k.mean(-1, keepdim=True)) / audio_16k.var(
        -1, keepdim=True, unbiased=False
    ).add(1e-7).sqrt()


def _speaker_and_content(wavlm, ecapa, audio_16k):
    normalized = _wavlm_inputs(audio_16k)
    speaker = _normalize_embedding(wavlm(normalized).embeddings)
    content = wavlm.wavlm(normalized).last_hidden_state
    ecapa_embedding = _normalize_embedding(ecapa.encode_batch(audio_16k))
    return speaker, ecapa_embedding, content


def _centroids(wavlm, ecapa, reference_paths, device):
    import torchaudio

    wavlm_values, ecapa_values = [], []
    with torch.no_grad():
        for path in reference_paths:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.mean(1)).to(device)[None]
            waveform = torchaudio.functional.resample(waveform, rate, 16_000)
            wavlm_value, ecapa_value, _ = _speaker_and_content(wavlm, ecapa, waveform)
            wavlm_values.append(wavlm_value)
            ecapa_values.append(ecapa_value)
    return (
        _normalize_embedding(torch.stack(wavlm_values).mean(0)),
        _normalize_embedding(torch.stack(ecapa_values).mean(0)),
    )


def _source_crop(source_path, extractor, load_wav, sample_rate: int, seconds: float, device):
    full_f0 = np.asarray(extractor.process(source_path, verbose=False), dtype=np.float32)
    voiced = np.flatnonzero(full_f0 > 0)
    if not len(voiced):
        raise RuntimeError("source crop contains no RMVPE voiced frame")
    frames = int(round(seconds * sample_rate / 480))
    start_frame = max(0, min(int(voiced[0]), len(full_f0) - frames))
    waveform = load_wav(source_path, sample_rate).to(device)
    start_sample = start_frame * 480
    samples = int(round(seconds * sample_rate))
    crop = waveform[:, start_sample:start_sample + samples]
    if crop.shape[-1] != samples:
        raise RuntimeError("source is shorter than the fixed crop")
    f0 = torch.from_numpy(full_f0[start_frame:start_frame + frames])[None].to(device)
    return crop, f0, start_frame


def run_feasibility(args):
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoModelForAudioXVector
    from transformers.models.wavlm import modeling_wavlm

    _disable_optional_peft(modeling_wavlm)

    cache = Path(args.soulx_root)
    sys.path.insert(0, str(cache))
    from preprocess.tools.f0_extraction import F0Extractor
    from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
    from soulxsinger.utils.audio_utils import load_wav
    from soulxsinger.utils.file_utils import load_config

    sys.path.insert(0, str(Path.cwd() / "src"))
    from gyu_singer.inference.latent_adapter import SoulXRealLatentAdapters

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    model = SoulXSingerSVC(config).to(device).float().eval()
    state = torch.load(args.model, weights_only=False, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model.mel.float()
    _freeze(model)
    _freeze(model.whisper_encoder.model)

    saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapters = SoulXRealLatentAdapters(**saved["config"]).to(device).float().eval()
    adapters.load_state_dict(saved["model"])
    _freeze(adapters)
    for parameter in adapters.identity.parameters():
        parameter.requires_grad_(True)
    initial = {name: parameter.detach().clone() for name, parameter in adapters.identity.named_parameters()}
    identity = torch.from_numpy(np.load(args.identity).astype(np.float32)).to(device)

    extractor = F0Extractor(args.rmvpe, device="cuda", target_sr=24_000, hop_size=480, verbose=False)
    source, source_f0, start_frame = _source_crop(
        args.source, extractor, load_wav, config.audio.sample_rate, args.crop_seconds, device
    )
    reference = load_wav(args.reference, config.audio.sample_rate).to(device)
    reference_f0 = torch.from_numpy(np.asarray(extractor.process(args.reference, verbose=False), dtype=np.float32))[None].to(device)
    with torch.no_grad():
        prompt_mel, prompt_condition, target_condition = _features(
            model, reference, source, reference_f0, source_f0
        )

    def production(target):
        torch.manual_seed(args.seed)
        with torch.no_grad():
            return _decode(
                model, prompt_mel, prompt_condition, target,
                source.shape[-1], args.total_steps, args.cfg,
            )

    off_first = production(target_condition)
    off_second = production(target_condition)
    with torch.no_grad():
        current_target = adapters.identity(target_condition, identity)
    current = production(current_target)
    off_repeat_max = float((off_first - off_second).abs().max())

    wavlm = AutoModelForAudioXVector.from_pretrained(args.wavlm).to(device).eval()
    wavlm_state = torch.load(Path(args.wavlm) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    wavlm_load = wavlm.load_state_dict(_convert_legacy_wavlm_weight_norm(wavlm_state), strict=False)
    if wavlm_load.missing_keys or wavlm_load.unexpected_keys:
        raise RuntimeError(
            f"WavLM checkpoint conversion failed: missing={wavlm_load.missing_keys} "
            f"unexpected={wavlm_load.unexpected_keys}"
        )
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=args.ecapa,
        run_opts={"device": "cuda"},
    )
    _freeze(wavlm)
    _freeze(ecapa.mods)
    references = [f"data/processed/master/{index}.wav" for index in range(171, 195)]
    wavlm_centroid, ecapa_centroid = _centroids(wavlm, ecapa, references, device)
    off_16k = torchaudio.functional.resample(off_first[None], config.audio.sample_rate, 16_000)
    with torch.no_grad():
        _, _, off_content = _speaker_and_content(wavlm, ecapa, off_16k)

    for parameter in adapters.identity.parameters():
        parameter.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    adapted = adapters.identity(target_condition, identity)
    frozen_condition = model.cfm_decoder.model.cond_emb(
        torch.cat([prompt_condition, adapted.detach()], 1)
    ).detach()
    trainable_condition = model.cfm_decoder.model.cond_emb(
        torch.cat([prompt_condition, adapted], 1)
    )
    torch.manual_seed(args.seed)
    mel = truncated_reverse_diffusion(
        model.cfm_decoder.model, prompt_mel, frozen_condition, trainable_condition,
        args.total_steps, args.grad_steps, args.cfg,
    )
    candidate = model.vocoder(mel.transpose(1, 2)[0:1]).squeeze().float()
    candidate = candidate[:source.shape[-1]] if candidate.shape[-1] >= source.shape[-1] else F.pad(
        candidate, (0, source.shape[-1] - candidate.shape[-1])
    )
    initial_max = float((candidate.detach() - current).abs().max())
    initial_mean = float((candidate.detach() - current).abs().mean())
    initial_match = torch.allclose(candidate.detach(), current, atol=1e-5, rtol=1e-4)

    candidate_16k = torchaudio.functional.resample(candidate[None], config.audio.sample_rate, 16_000)
    candidate_wavlm, candidate_ecapa, candidate_content = _speaker_and_content(
        wavlm, ecapa, candidate_16k
    )
    speaker_wavlm = 1 - F.cosine_similarity(candidate_wavlm, wavlm_centroid).mean()
    speaker_ecapa = 1 - F.cosine_similarity(candidate_ecapa, ecapa_centroid).mean()
    preservation = preservation_losses(candidate[None], off_first[None], config.audio.sample_rate, source_f0[0])
    content_loss = F.mse_loss(candidate_content, off_content)
    output_regularization = F.mse_loss(adapted, target_condition)
    gate_regularization = (adapters.identity.gate - initial["gate"]).square()
    parameter_regularization = sum(
        (parameter - initial[name]).square().mean()
        for name, parameter in adapters.identity.named_parameters()
    )
    losses = {
        "wavlm_speaker": speaker_wavlm,
        "ecapa_speaker": speaker_ecapa,
        **preservation,
        "content": content_loss,
        "adapter_output": output_regularization,
        "gate": gate_regularization,
        "parameter_drift": parameter_regularization,
    }
    weights = {
        "wavlm_speaker": 1.0, "ecapa_speaker": 1.0, "waveform": 1.0,
        "stft_256": 0.1, "stft_1024": 0.1, "stft_4096": 0.1,
        "pitch_period": 0.1, "content": 0.1, "adapter_output": 0.01,
        "gate": 0.01, "parameter_drift": 0.01,
    }
    total_loss = sum(weights[name] * value for name, value in losses.items())
    finite_losses = bool(torch.isfinite(total_loss) and all(torch.isfinite(value) for value in losses.values()))
    if finite_losses and initial_match:
        total_loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    audit = gradient_audit(adapters.identity, {"soulx": model, "wavlm": wavlm, "ecapa": ecapa.mods})
    relative_drift = float(torch.sqrt(sum(
        (parameter.detach() - initial[name]).float().square().sum()
        for name, parameter in adapters.identity.named_parameters()
    )) / torch.sqrt(sum(value.float().square().sum() for value in initial.values())).clamp_min(1e-12))

    paths = {}
    for name, audio in {"identity_off": off_first, "current_v07": current, "truncated": candidate}.items():
        path = output / f"{name}.wav"
        sf.write(path, audio.detach().cpu().numpy(), config.audio.sample_rate, subtype="PCM_24")
        paths[name] = str(path)
    gates = {
        "identity_off_reproduced": off_repeat_max == 0.0,
        "initial_v07_match": bool(initial_match),
        "finite_audio": bool(torch.isfinite(candidate).all()),
        "finite_losses": finite_losses,
        "adapter_gradient": bool(audit["adapter_gradient_finite"] and audit["adapter_gradient_nonzero"]),
        "no_frozen_gradients": not audit["unexpected_frozen_gradients"],
        "no_parameter_update": relative_drift == 0.0,
    }
    report = {
        "status": "feasibility_pass" if all(gates.values()) else "diagnostic_reject",
        "feasibility_pass": all(gates.values()),
        "grad_steps": args.grad_steps,
        "total_steps": args.total_steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "crop": {
            "source": args.source,
            "start_seconds": round(start_frame * 480 / config.audio.sample_rate, 4),
            "duration_seconds": args.crop_seconds,
        },
        "baseline": {
            "identity_off_repeat_max_abs": off_repeat_max,
            "current_v07_candidate_max_abs": initial_max,
            "current_v07_candidate_mean_abs": initial_mean,
            "allclose_atol": 1e-5,
            "allclose_rtol": 1e-4,
        },
        "losses": {name: float(value.detach()) for name, value in losses.items()} | {"total": float(total_loss.detach())},
        "loss_weights": weights,
        "gradient_audit": audit,
        "relative_parameter_drift": relative_drift,
        "runtime_seconds": round(elapsed, 4),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gates": gates,
        "paths": paths,
        "revisions": {
            "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "soulx": "81aeb3ae772c70093c3de74dc23c92d983801ae4",
            "adapter": args.adapter,
        },
        "wavlm_legacy_weight_norm_restored": True,
        "optimizer_steps": 0,
        "runtime_integration": False,
    }
    (output / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    root = Path.cwd()
    soulx = root / "data/cache/soulx-singer"
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--crop-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--soulx-root", default=str(soulx))
    parser.add_argument("--model", default=str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"))
    parser.add_argument("--config", default=str(soulx / "soulxsinger/config/soulxsinger.yaml"))
    parser.add_argument("--rmvpe", default=str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"))
    parser.add_argument("--adapter", default="checkpoints/gyu_real_latent_adapters_v0.7.pt")
    parser.add_argument("--identity", default="artifacts/reports/rc8_sustained_decode_sweep/fixed_identity.npy")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    parser.add_argument("--source", default="artifacts/reports/rc8_sustained_decode_sweep/omnivoice_source.wav")
    parser.add_argument("--wavlm", default="data/cache/wavlm-base-plus-sv")
    parser.add_argument("--ecapa", default="data/cache/spkrec-ecapa-voxceleb")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    try:
        report = run_feasibility(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "diagnostic_reject",
            "feasibility_pass": False,
            "grad_steps": args.grad_steps,
            "error_type": type(error).__name__,
            "error": str(error),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_steps": 0,
            "runtime_integration": False,
        }
        (output / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["feasibility_pass"] else 1)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
