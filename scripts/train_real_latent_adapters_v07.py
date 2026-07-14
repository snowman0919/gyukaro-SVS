#!/usr/bin/env python3
"""Train separable identity/style FiLM paths on captured SoulX decoder latents."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gyu_singer.data import acoustic_reference_features
from gyu_singer.inference.latent_adapter import SoulXRealLatentAdapters
from gyu_singer.model import MultiTeacherIdentityEncoder


STYLES = ("neutral", "soft", "breathy", "energetic", "dark", "bright")


def load_hidden(row: dict, device: str) -> torch.Tensor:
    return torch.load(row["latent_path"], map_location=device, weights_only=True).float()


def centroid(rows: list[dict], device: str) -> torch.Tensor:
    return torch.stack([load_hidden(row, device).mean((0, 1)) for row in rows]).mean(0)


def proxy_axis(rows: list[dict], key: str, device: str) -> tuple[torch.Tensor, dict]:
    """Ridge direction learned from real latent/audio pairs on train only."""
    x = torch.stack([load_hidden(row, device).mean((0, 1)) for row in rows])
    y = torch.tensor([np.log(max(row["audio_acoustic_proxies_inferred"][key], 1e-12)) for row in rows], device=device, dtype=x.dtype)
    x, y = x - x.mean(0), (y - y.mean()) / y.std().clamp_min(1e-6)
    # Dual ridge is stable for 36 rows x 512 latent channels.
    axis = x.T @ torch.linalg.solve(x @ x.T + 1e-2 * torch.eye(len(x), device=device), y)
    axis = axis / axis.square().mean().sqrt().clamp_min(1e-6)
    return axis, {"train_rows": len(rows), "target": f"log({key})", "ridge": 0.01}


def proxy_correlations(rows: list[dict], axes: dict[str, torch.Tensor], center: torch.Tensor, device: str) -> dict[str, float | None]:
    result = {}
    for name, axis in axes.items():
        predicted = [float(torch.dot(load_hidden(row, device).mean((0, 1)) - center, axis)) for row in rows]
        key = {"brightness": "spectral_centroid_hz", "energy": "rms", "breathiness": "high_frequency_ratio_4khz"}[name]
        target = [np.log(max(row["audio_acoustic_proxies_inferred"][key], 1e-12)) for row in rows]
        result[name] = round(float(np.corrcoef(predicted, target)[0, 1]), 5) if len(rows) > 2 else None
    return result


def main() -> None:
    torch.manual_seed(707)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(line) for line in Path("data/manifests/soulx_real_latents_v07.jsonl").read_text().splitlines() if line]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    real = [row for row in train if row["source_type"] == "real_gyu"]
    teacher_neutral = [row for row in train if row["source_type"] == "teacher_style" and row["style"] == "neutral"]
    if not real or not teacher_neutral:
        raise RuntimeError("real GYU and teacher-neutral latents are required")

    real_center, neutral_center = centroid(real, device), centroid(teacher_neutral, device)
    identity_direction = real_center - neutral_center
    empirical_rms = {}
    for style in STYLES[1:]:
        evidence = [row for row in train if row["source_type"] == "teacher_style" and row["style"] == style]
        if not evidence:
            raise RuntimeError(f"missing real-latent style evidence: {style}")
        empirical_rms[style] = float((centroid(evidence, device) - neutral_center).square().mean().sqrt())
    teacher_train = [row for row in train if row["source_type"] == "teacher_style"]
    axes = {}
    axes["brightness"], proxy_fit = proxy_axis(teacher_train, "spectral_centroid_hz", device)
    axes["energy"], _ = proxy_axis(teacher_train, "rms", device)
    axes["breathiness"], _ = proxy_axis(teacher_train, "high_frequency_ratio_4khz", device)
    style_directions = {
        "neutral": torch.zeros_like(neutral_center),
        "soft": -axes["energy"] * empirical_rms["soft"],
        "breathy": axes["breathiness"] * empirical_rms["breathy"],
        "energetic": axes["energy"] * empirical_rms["energetic"],
        "dark": -axes["brightness"] * empirical_rms["dark"],
        "bright": axes["brightness"] * empirical_rms["bright"],
    }

    saved = torch.load("checkpoints/gyu_identity_space_v0.6.pt", map_location="cpu", weights_only=False)
    encoder = MultiTeacherIdentityEncoder(**saved["model_config"]).eval(); encoder.load_state_dict(saved["model"])
    with torch.inference_mode():
        identity = encoder.student(acoustic_reference_features("data/processed/master/216.wav")[None])
    identity = identity.detach().clone().to(device)

    model = SoulXRealLatentAdapters().to(device).train()
    identity_optimizer = torch.optim.AdamW(model.identity.parameters(), lr=2e-3, weight_decay=1e-4)
    style_optimizer = torch.optim.AdamW(model.style.parameters(), lr=2e-3, weight_decay=1e-4)
    neutral_inputs = [row for row in train if row["style"] == "neutral"]
    history = []
    for step in range(1, 801):
        row = train[(step - 1) % len(train)]
        hidden = load_hidden(row, device)
        target_direction = identity_direction if row["source_type"] == "teacher_style" else torch.zeros_like(identity_direction)
        identity_output = model.identity(hidden, identity)
        identity_loss = float(row["trust_weight"]) * F.mse_loss(identity_output - hidden, target_direction[None, None].expand_as(hidden))
        identity_optimizer.zero_grad(); identity_loss.backward(); identity_optimizer.step()

        neutral_row = neutral_inputs[(step - 1) % len(neutral_inputs)]
        neutral_hidden = load_hidden(neutral_row, device)
        style = STYLES[(step - 1) % len(STYLES)]
        style_condition = torch.zeros(1, 64, device=device); style_condition[0, STYLES.index(style)] = 1
        style_output = model.style(neutral_hidden, style_condition)
        style_loss = float(neutral_row["trust_weight"]) * F.mse_loss(style_output - neutral_hidden, style_directions[style][None, None].expand_as(neutral_hidden))
        style_optimizer.zero_grad(); style_loss.backward(); style_optimizer.step()
        if step % 200 == 0:
            history.append({"step": step, "identity_loss": round(float(identity_loss.detach()), 7), "style_loss": round(float(style_loss.detach()), 7), "identity_gate": round(float(torch.sigmoid(model.identity.gate).detach()), 5), "style_gate": round(float(torch.sigmoid(model.style.gate).detach()), 5)})

    model.eval()
    validation_identity, validation_style = [], []
    with torch.inference_mode():
        for row in validation:
            hidden = load_hidden(row, device)
            target = identity_direction if row["source_type"] == "teacher_style" else torch.zeros_like(identity_direction)
            validation_identity.append(float(F.mse_loss(model.identity(hidden, identity) - hidden, target[None, None].expand_as(hidden))))
            style_condition = torch.zeros(1, 64, device=device); style_condition[0, STYLES.index(row["style"])] = 1
            validation_style.append(float(F.mse_loss(model.style(hidden, style_condition) - hidden, style_directions[row["style"]][None, None].expand_as(hidden))))

    checkpoint = Path("checkpoints/gyu_real_latent_adapters_v0.7.pt")
    torch.save({"model": model.cpu().state_dict(), "config": {"identity_dim": 64, "style_dim": 64, "hidden_dim": 512}, "version": "v0.7", "latent_source": "actual SoulXSingerSVC.infer_segment.gt_decoder_inp", "identity_space": "checkpoints/gyu_identity_space_v0.6.pt", "styles": list(STYLES), "rows": len(rows)}, checkpoint)
    teacher_validation = [row for row in validation if row["source_type"] == "teacher_style"]
    teacher_test = [row for row in rows if row["split"] == "test" and row["source_type"] == "teacher_style"]
    report = {"rows": len(rows), "train_rows": len(train), "validation_rows": len(validation), "test_rows": sum(row["split"] == "test" for row in rows), "identity_adapter_parameters": sum(parameter.numel() for parameter in model.identity.parameters()), "style_adapter_parameters": sum(parameter.numel() for parameter in model.style.parameters()), "identity_direction_rms": round(float(identity_direction.square().mean().sqrt()), 6), "style_direction_rms": {style: round(float(direction.square().mean().sqrt()), 6) for style, direction in style_directions.items()}, "style_supervision": "ridge axes from measured source-audio proxies on train real SoulX latents", "proxy_fit": proxy_fit, "proxy_axis_validation_correlations": proxy_correlations(teacher_validation, axes, neutral_center, device), "proxy_axis_test_correlations": proxy_correlations(teacher_test, axes, neutral_center, device), "validation_identity_mse": round(sum(validation_identity) / len(validation_identity), 7), "validation_style_mse": round(sum(validation_style) / len(validation_style), 7), "history": history, "checkpoint": str(checkpoint), "backbone_frozen": True, "dummy_latents_used": False}
    Path("artifacts/reports/real_latent_training_v07.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
