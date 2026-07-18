#!/usr/bin/env python3
"""Train one bounded K=2/K=4 final-WAV identity diagnostic candidate."""
from __future__ import annotations

import argparse
import json
import math
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


FIXED_SEEDS = (7, 21, 42)
LOSS_WEIGHTS = {
    "wavlm_speaker": 1.0,
    "ecapa_speaker": 1.0,
    "waveform": 1.0,
    "stft_256": 0.1,
    "stft_1024": 0.1,
    "stft_4096": 0.1,
    "pitch_period": 0.1,
    "content": 0.1,
    "adapter_output": 0.01,
    "gate": 0.01,
    "parameter_drift": 0.01,
}


def configure_identity_only(adapters: nn.Module) -> list[nn.Parameter]:
    for parameter in adapters.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    selected = list(adapters.identity.parameters())
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def clone_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _global_norm(values) -> torch.Tensor:
    values = list(values)
    if not values:
        return torch.tensor(0.0)
    return torch.sqrt(sum(value.detach().float().square().sum() for value in values))


def gradient_safety(
    adapter: nn.Module,
    frozen_modules: dict[str, nn.Module],
    max_gradient_norm: float = 1.0,
) -> dict:
    unexpected = [
        f"{module_name}.{parameter_name}"
        for module_name, module in frozen_modules.items()
        for parameter_name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
    finite = bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)
    norm = float(_global_norm(gradients)) if gradients else 0.0
    reason = None
    if unexpected:
        reason = "unexpected_frozen_gradient"
    elif not finite:
        reason = "nonfinite_adapter_gradient" if gradients else "zero_adapter_gradient"
    elif norm == 0.0:
        reason = "zero_adapter_gradient"
    elif norm > max_gradient_norm:
        reason = "gradient_norm_limit"
    return {
        "pass": reason is None,
        "reason": reason,
        "adapter_gradient_norm": norm,
        "adapter_gradient_finite": finite,
        "unexpected_frozen_gradients": unexpected,
    }


def update_safety(
    adapter: nn.Module,
    before_step: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    max_relative_update: float = 0.005,
    max_relative_drift: float = 0.05,
) -> dict:
    current = dict(adapter.named_parameters())
    update_norm = _global_norm(current[name] - before_step[name] for name in current)
    before_norm = _global_norm(before_step.values()).clamp_min(1e-12)
    drift_norm = _global_norm(current[name] - initial[name] for name in current)
    initial_norm = _global_norm(initial.values()).clamp_min(1e-12)
    relative_update = float(update_norm / before_norm)
    relative_drift = float(drift_norm / initial_norm)
    finite = math.isfinite(relative_update) and math.isfinite(relative_drift)
    reason = None
    if not finite:
        reason = "nonfinite_parameter_update"
    elif relative_update > max_relative_update:
        reason = "relative_update_limit"
    elif relative_drift > max_relative_drift:
        reason = "relative_drift_limit"
    return {
        "pass": reason is None,
        "reason": reason,
        "update_norm": float(update_norm),
        "relative_update": relative_update,
        "relative_drift": relative_drift,
    }


def _load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _warp_content(content: torch.Tensor, warp: np.ndarray | None, strength: float, frames: int):
    if warp is None:
        return F.interpolate(content.transpose(1, 2), size=frames, mode="linear", align_corners=False).transpose(1, 2)
    positions = torch.from_numpy(warp.astype(np.float32)).to(content.device) * max(content.shape[1] - 1, 1)
    if len(positions) != frames:
        positions = F.interpolate(positions[None, None], size=frames, mode="linear", align_corners=False)[0, 0]
    left = positions.floor().long().clamp(0, content.shape[1] - 1)
    right = (left + 1).clamp(0, content.shape[1] - 1)
    fraction = (positions - left).to(content.dtype)[None, :, None]
    warped = content[:, left, :] * (1 - fraction) + content[:, right, :] * fraction
    if strength >= 1:
        return warped
    base = F.interpolate(content.transpose(1, 2), size=frames, mode="linear", align_corners=False).transpose(1, 2)
    return base * (1 - strength) + warped * strength


def _features_with_warp(model, reference, source, reference_f0, target_f0, warp, strength):
    prompt_mel = model.mel(reference.float())
    prompt_pitch = model.f0_to_coarse(reference_f0)
    target_pitch = model.f0_to_coarse(target_f0)
    prompt_pitch = F.pad(prompt_pitch, (0, max(0, prompt_mel.shape[1] - prompt_pitch.shape[1])))
    prompt_pitch = prompt_pitch[:, :prompt_mel.shape[1]]
    prompt_content = model.whisper_encoder.encode(reference, sr=model.audio_cfg.sample_rate)
    target_content = model.whisper_encoder.encode(source, sr=model.audio_cfg.sample_rate)
    prompt_content = F.pad(prompt_content, (0, 0, 0, max(0, prompt_pitch.shape[1] - prompt_content.shape[1])))
    target_content = _warp_content(target_content, warp, strength, target_pitch.shape[1])
    content = torch.cat([prompt_content[:, :prompt_pitch.shape[1]], target_content], 1)
    pitch = torch.cat([prompt_pitch, target_pitch], 1)
    encoded = content + model.f0_encoder(pitch)
    return prompt_mel, encoded[:, :prompt_mel.shape[1]], encoded[:, prompt_mel.shape[1]:]


def _crop_bounds(target_f0: np.ndarray, duration: float, frame_rate: int = 50) -> tuple[int, int]:
    if 2.0 <= duration <= 4.0:
        return 0, len(target_f0)
    frames = min(len(target_f0), 3 * frame_rate)
    voiced = np.flatnonzero(target_f0 > 0)
    if not len(voiced):
        raise RuntimeError("target F0 contains no voiced frame")
    start = max(0, min(int(voiced[0]), len(target_f0) - frames))
    return start, start + frames


def _decode_full(model, prompt_mel, prompt_condition, target_condition, samples, steps, cfg, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        mel = model.cfm_decoder.reverse_diffusion(
            prompt_mel, prompt_condition, target_condition, n_timesteps=steps, cfg=cfg
        )
        audio = model.vocoder(mel.transpose(1, 2)[0:1]).squeeze().float()
    return audio[:samples] if audio.shape[-1] >= samples else F.pad(audio, (0, samples - audio.shape[-1]))


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def run_training(args) -> dict:
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoModelForAudioXVector
    from transformers.models.wavlm import modeling_wavlm

    sys.path[:0] = [str(Path(args.soulx_root)), str(Path.cwd() / "src"), str(Path.cwd() / "scripts")]
    from preprocess.tools.f0_extraction import F0Extractor
    from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
    from soulxsinger.utils.audio_utils import load_wav
    from soulxsinger.utils.file_utils import load_config
    from gyu_singer.inference.latent_adapter import SoulXRealLatentAdapters
    from probe_truncated_identity_grad import (
        _centroids,
        _convert_legacy_wavlm_weight_norm,
        _disable_optional_peft,
        _freeze,
        _speaker_and_content,
        preservation_losses,
        truncated_reverse_diffusion,
    )

    _disable_optional_peft(modeling_wavlm)
    root = Path.cwd()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "identity_adapter_diagnostic.pt"
    checkpoint.unlink(missing_ok=True)
    if args.total_steps != 64 or not 1 <= args.epochs <= 2:
        raise RuntimeError("bounded diagnostic requires exactly 64 diffusion steps and at most two epochs")
    rows = _load_manifest(Path(args.manifest))
    train_rows = [row for row in rows if row["split"] == "train" and row["identity_training_eligible"]]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if [row["id"] for row in train_rows] != ["korean", "english", "japanese"]:
        raise RuntimeError("fixed training split changed")
    if any(row["id"] in {"heldout_ja", "review_rapid_ko"} for row in train_rows + validation_rows):
        raise RuntimeError("excluded/protected phrase entered optimization")

    device = torch.device("cuda")
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
    selected_parameters = configure_identity_only(adapters)
    initial = clone_parameters(adapters.identity)
    initial_full_state = _state_dict_cpu(adapters)

    extractor = F0Extractor(args.rmvpe, device="cuda", target_sr=24_000, hop_size=480, verbose=False)
    reference = load_wav(args.reference, config.audio.sample_rate).to(device)
    reference_f0 = torch.from_numpy(
        np.asarray(extractor.process(args.reference, verbose=False), dtype=np.float32)
    )[None].to(device)

    def prepare(row):
        source = load_wav(str(root / row["source_path"]), config.audio.sample_rate).to(device)
        target_f0_np = np.load(root / row["f0_path"]).astype(np.float32)
        expected_frames = round(source.shape[-1] / 480)
        if len(target_f0_np) != expected_frames:
            target_f0_np = np.interp(
                np.arange(expected_frames), np.linspace(0, expected_frames - 1, len(target_f0_np)), target_f0_np
            ).astype(np.float32)
        warp = np.load(root / row["content_warp_path"]).astype(np.float32) if row.get("content_warp_path") else None
        with torch.no_grad():
            prompt_mel, prompt_condition, target_condition = _features_with_warp(
                model, reference, source, reference_f0,
                torch.from_numpy(target_f0_np)[None].to(device), warp,
                float(row.get("content_warp_strength", 0.0)),
            )
        start, end = _crop_bounds(target_f0_np, source.shape[-1] / config.audio.sample_rate)
        frames = end - start
        return {
            "row": row,
            "prompt_mel": prompt_mel.detach(),
            "prompt_condition": prompt_condition.detach(),
            "target_condition": target_condition[:, start:end].detach(),
            "target_f0": torch.from_numpy(target_f0_np[start:end]).to(device),
            "samples": frames * 480,
            "crop_start_frame": start,
            "crop_frames": frames,
        }

    prepared = {row["id"]: prepare(row) for row in train_rows + validation_rows}
    wavlm = AutoModelForAudioXVector.from_pretrained(args.wavlm).to(device).eval()
    wavlm_state = torch.load(Path(args.wavlm) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    load_result = wavlm.load_state_dict(_convert_legacy_wavlm_weight_norm(wavlm_state), strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"WavLM checkpoint conversion failed: {load_result}")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=args.ecapa,
        run_opts={"device": "cuda"},
    )
    _freeze(wavlm)
    _freeze(ecapa.mods)
    references = [root / f"data/processed/master/{index}.wav" for index in range(171, 195)]
    wavlm_centroid, ecapa_centroid = _centroids(wavlm, ecapa, references, device)

    baseline_cache = {}
    for row in train_rows + validation_rows:
        item = prepared[row["id"]]
        seeds = FIXED_SEEDS if row["split"] == "train" else (21,)
        for seed in seeds:
            off = _decode_full(
                model, item["prompt_mel"], item["prompt_condition"], item["target_condition"],
                item["samples"], args.total_steps, float(row["cfg"]), seed,
            )
            off_16k = torchaudio.functional.resample(off[None], config.audio.sample_rate, 16_000)
            with torch.no_grad():
                _, _, content = _speaker_and_content(wavlm, ecapa, off_16k)
            baseline_cache[(row["id"], seed)] = (off.detach(), content.detach())

    def losses_for(item, seed, truncated):
        row = item["row"]
        baseline, baseline_content = baseline_cache[(row["id"], seed)]
        adapted = adapters.identity(item["target_condition"], torch.from_numpy(
            np.load(root / row["identity_path"]).astype(np.float32)
        ).to(device))
        if truncated:
            frozen_condition = model.cfm_decoder.model.cond_emb(
                torch.cat([item["prompt_condition"], adapted.detach()], 1)
            ).detach()
            trainable_condition = model.cfm_decoder.model.cond_emb(
                torch.cat([item["prompt_condition"], adapted], 1)
            )
            torch.manual_seed(seed)
            mel = truncated_reverse_diffusion(
                model.cfm_decoder.model, item["prompt_mel"], frozen_condition,
                trainable_condition, args.total_steps, args.grad_steps, float(row["cfg"]),
            )
            candidate = model.vocoder(mel.transpose(1, 2)[0:1]).squeeze().float()
            candidate = candidate[:item["samples"]] if candidate.shape[-1] >= item["samples"] else F.pad(
                candidate, (0, item["samples"] - candidate.shape[-1])
            )
        else:
            candidate = _decode_full(
                model, item["prompt_mel"], item["prompt_condition"], adapted,
                item["samples"], args.total_steps, float(row["cfg"]), seed,
            )
        candidate_16k = torchaudio.functional.resample(candidate[None], config.audio.sample_rate, 16_000)
        speaker_wavlm, speaker_ecapa, candidate_content = _speaker_and_content(wavlm, ecapa, candidate_16k)
        preservation = preservation_losses(
            candidate[None], baseline[None], config.audio.sample_rate, item["target_f0"]
        )
        values = {
            "wavlm_speaker": 1 - F.cosine_similarity(speaker_wavlm, wavlm_centroid).mean(),
            "ecapa_speaker": 1 - F.cosine_similarity(speaker_ecapa, ecapa_centroid).mean(),
            **preservation,
            "content": F.mse_loss(candidate_content, baseline_content),
            "adapter_output": F.mse_loss(adapted, item["target_condition"]),
            "gate": (adapters.identity.gate - initial["gate"]).square(),
            "parameter_drift": sum(
                (parameter - initial[name]).square().mean()
                for name, parameter in adapters.identity.named_parameters()
            ),
        }
        total = sum(LOSS_WEIGHTS[name] * value for name, value in values.items())
        return total, values, candidate

    optimizer = torch.optim.AdamW(selected_parameters, lr=args.learning_rate, weight_decay=0.0)
    history, validation_history, epoch_states = [], [], []
    started = time.perf_counter()
    peak_allocated = peak_reserved = 0
    for epoch in range(1, args.epochs + 1):
        adapters.train()
        adapters.style.eval()
        for row in train_rows:
            for seed in FIXED_SEEDS:
                item = prepared[row["id"]]
                optimizer.zero_grad(set_to_none=True)
                before = clone_parameters(adapters.identity)
                torch.cuda.reset_peak_memory_stats()
                step_started = time.perf_counter()
                total, values, _ = losses_for(item, seed, truncated=True)
                if not bool(torch.isfinite(total)) or not all(bool(torch.isfinite(value)) for value in values.values()):
                    raise RuntimeError(f"nonfinite_loss:{row['id']}:{seed}")
                total.backward()
                pre_clip = gradient_safety(
                    adapters.identity, {"soulx": model, "wavlm": wavlm, "ecapa": ecapa.mods, "style": adapters.style}
                )
                if not pre_clip["pass"]:
                    raise RuntimeError(f"gradient_reject:{pre_clip['reason']}:{row['id']}:{seed}")
                torch.nn.utils.clip_grad_norm_(selected_parameters, args.clip_norm)
                post_clip_norm = float(_global_norm(
                    parameter.grad for parameter in selected_parameters if parameter.grad is not None
                ))
                optimizer.step()
                update = update_safety(adapters.identity, before, initial)
                if not update["pass"]:
                    raise RuntimeError(f"update_reject:{update['reason']}:{row['id']}:{seed}")
                torch.cuda.synchronize()
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                record = {
                    "epoch": epoch, "step": len(history) + 1, "phrase": row["id"], "seed": seed,
                    "loss": float(total.detach()),
                    "losses": {name: float(value.detach()) for name, value in values.items()},
                    "pre_clip_gradient_norm": pre_clip["adapter_gradient_norm"],
                    "post_clip_gradient_norm": post_clip_norm,
                    **update,
                    "runtime_seconds": round(time.perf_counter() - step_started, 4),
                }
                history.append(record)
                print(json.dumps({key: record[key] for key in (
                    "epoch", "step", "phrase", "seed", "loss", "pre_clip_gradient_norm",
                    "post_clip_gradient_norm", "relative_update", "relative_drift", "runtime_seconds",
                )}), flush=True)

        adapters.eval()
        validation_rows_result = []
        with torch.no_grad():
            for row in validation_rows:
                total, values, _ = losses_for(prepared[row["id"]], 21, truncated=False)
                validation_rows_result.append({
                    "phrase": row["id"], "seed": 21, "loss": float(total),
                    "losses": {name: float(value) for name, value in values.items()},
                })
        combined = float(np.mean([row["loss"] for row in validation_rows_result]))
        validation_history.append({"epoch": epoch, "combined_loss": combined, "rows": validation_rows_result})
        epoch_states.append(_state_dict_cpu(adapters))
        print(json.dumps({"epoch": epoch, "validation_combined_loss": combined}), flush=True)

    selected_index = int(np.argmin([row["combined_loss"] for row in validation_history]))
    selected_epoch = selected_index + 1
    adapters.load_state_dict(epoch_states[selected_index])
    final_update = update_safety(adapters.identity, clone_parameters(adapters.identity), initial)
    if not final_update["pass"]:
        raise RuntimeError(f"selected_checkpoint_reject:{final_update['reason']}")
    torch.save({
        "model": _state_dict_cpu(adapters), "config": saved["config"], "version": "v0.7",
        "diagnostic": {"grad_steps": args.grad_steps, "total_steps": args.total_steps,
                       "selected_epoch": selected_epoch, "runtime_integration": False},
    }, checkpoint)
    final_full_state = _state_dict_cpu(adapters)
    initial_unchanged = all(
        torch.equal(value, final_full_state[name])
        for name, value in initial_full_state.items() if not name.startswith("identity.")
    )
    report = {
        "status": "training_pass",
        "grad_steps": args.grad_steps,
        "total_steps": args.total_steps,
        "optimizer_steps": len(history),
        "fixed_seeds": list(FIXED_SEEDS),
        "train_phrases": [row["id"] for row in train_rows],
        "validation_phrases": [row["id"] for row in validation_rows],
        "excluded": ["heldout_ja"],
        "protected_evaluation_only": ["review_rapid_ko"],
        "history": history,
        "validation": validation_history,
        "selected_epoch": selected_epoch,
        "selected_validation_loss": validation_history[selected_index]["combined_loss"],
        "selected_parameter_safety": final_update,
        "non_identity_state_unchanged": initial_unchanged,
        "loss_weights": LOSS_WEIGHTS,
        "checkpoint": str(checkpoint),
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "runtime_integration": False,
        "revisions": {
            "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "soulx": "81aeb3ae772c70093c3de74dc23c92d983801ae4",
            "base_adapter": args.adapter,
        },
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    root = Path.cwd()
    soulx = root / "data/cache/soulx-singer"
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="data/manifests/truncated_identity_diagnostic.jsonl")
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=0.1)
    parser.add_argument("--soulx-root", default=str(soulx))
    parser.add_argument("--model", default=str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"))
    parser.add_argument("--config", default=str(soulx / "soulxsinger/config/soulxsinger.yaml"))
    parser.add_argument("--rmvpe", default=str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"))
    parser.add_argument("--adapter", default="checkpoints/gyu_real_latent_adapters_v0.7.pt")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    parser.add_argument("--wavlm", default="data/cache/wavlm-base-plus-sv")
    parser.add_argument("--ecapa", default="data/cache/spkrec-ecapa-voxceleb")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = run_training(args)
    except Exception as error:
        (output / "identity_adapter_diagnostic.pt").unlink(missing_ok=True)
        report = {
            "status": "diagnostic_reject", "grad_steps": args.grad_steps,
            "error_type": type(error).__name__, "error": str(error),
            "checkpoint_saved": False, "runtime_integration": False,
        }
        (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    print(json.dumps({
        "status": report["status"], "grad_steps": report["grad_steps"],
        "optimizer_steps": report["optimizer_steps"], "selected_epoch": report["selected_epoch"],
        "checkpoint": report["checkpoint"],
    }, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
