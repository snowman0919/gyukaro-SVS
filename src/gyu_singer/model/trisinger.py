"""Compact phrase-level conditional-flow singing acoustic model."""
from __future__ import annotations

import torch
from torch import nn

from gyu_singer.frontend import FEATURE_SIZE


class UnifiedPhonemeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.embedding = nn.Embedding(2048, dim)
    def forward(self, phonemes: torch.Tensor) -> torch.Tensor: return self.embedding(phonemes)


class LanguageFeatureEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.language = nn.Embedding(3, dim); self.features = nn.Linear(FEATURE_SIZE, dim)
    def forward(self, language_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor: return self.language(language_ids) + self.features(features)


class ScoreEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.projection = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, midi: torch.Tensor, note_index: torch.Tensor, note_onset: torch.Tensor, note_duration: torch.Tensor, boundary: torch.Tensor, length: int) -> torch.Tensor:
        index = note_index.float() / max(1, length - 1)
        return self.projection(torch.stack((midi / 127.0, index, note_onset, note_duration, boundary, 1.0 - boundary), dim=-1))


class BlurredBoundaryEncoder(nn.Module):
    """TCSinger-inspired context sharing across hard note/phoneme boundaries."""
    def __init__(self, dim: int):
        super().__init__(); self.context = nn.Conv1d(dim, dim, 5, padding=2); self.mix = nn.Parameter(torch.tensor(0.35))
    def forward(self, content: torch.Tensor, score: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        local = self.context((content + score).transpose(1, 2)).transpose(1, 2)
        soft = torch.sigmoid(self.mix) * (1.0 - boundary.unsqueeze(-1) * 0.5)
        return (content + score) * (1.0 - soft) + local * soft


class TimbreEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.network = nn.Sequential(nn.Linear(160, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, reference_features: torch.Tensor) -> torch.Tensor: return self.network(reference_features)


class StyleEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.preset = nn.Embedding(8, dim); self.controls = nn.Linear(5, dim)
    def forward(self, preset: torch.Tensor, controls: torch.Tensor) -> torch.Tensor: return self.preset(preset) + self.controls(controls)


class PitchConditionEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.network = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, f0_hz: torch.Tensor, voiced: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.network(torch.stack((f0_hz.clamp_min(1).log(), voiced, residual), dim=-1))


class ConditionalFlowTransformer(nn.Module):
    def __init__(self, dim: int, latent_dim: int, layers: int = 3):
        super().__init__()
        self.noisy = nn.Linear(latent_dim, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.velocity = nn.Linear(dim, latent_dim)
    def forward(self, noisy_latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.noisy(noisy_latent) + condition + self.time(time[:, None, None]).expand_as(condition)
        return self.velocity(self.transformer(x))


class SingingDecoder(nn.Module):
    """Learned acoustic-latent projection; frozen MOSS codec decoder is used at render time."""
    def __init__(self, dim: int, latent_dim: int):
        super().__init__(); self.projection = nn.Linear(dim, latent_dim)
    def forward(self, condition: torch.Tensor) -> torch.Tensor: return self.projection(condition)


class TriSingerModel(nn.Module):
    def __init__(self, dim: int = 192, latent_dim: int = 768):
        super().__init__()
        self.latent_dim = latent_dim
        self.phoneme_encoder = UnifiedPhonemeEncoder(dim)
        self.language_encoder = LanguageFeatureEncoder(dim)
        self.score_encoder = ScoreEncoder(dim)
        self.blurred_boundary_encoder = BlurredBoundaryEncoder(dim)
        self.timbre_encoder = TimbreEncoder(dim)
        self.style_encoder = StyleEncoder(dim)
        self.pitch_encoder = PitchConditionEncoder(dim)
        self.conditional_flow_transformer = ConditionalFlowTransformer(dim, latent_dim)
        self.singing_decoder = SingingDecoder(dim, latent_dim)
        self.pitch_head = nn.Linear(dim, 1)
        self.distillation_head = nn.Linear(dim, 160)

    def condition(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content = self.phoneme_encoder(batch["phoneme_ids"]) + self.language_encoder(batch["language_ids"], batch["features"])
        score = self.score_encoder(batch["midi"], batch["note_index"], batch["note_onset"], batch["note_duration"], batch["boundary"], content.shape[1])
        boundary = self.blurred_boundary_encoder(content, score, batch["boundary"])
        timbre = self.timbre_encoder(batch["reference_features"])
        style = self.style_encoder(batch["style_preset"], batch["style_controls"])
        pitch = self.pitch_encoder(batch["f0_hz"], batch["voiced"], batch["residual"])
        condition = boundary + pitch + timbre[:, None, :] + style[:, None, :]
        return condition, timbre, content

    def forward(self, noisy_latent: torch.Tensor, time: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition, timbre, content = self.condition(batch)
        return {
            "velocity": self.conditional_flow_transformer(noisy_latent, time, condition),
            "acoustic_bias": self.singing_decoder(condition),
            "pitch_log_f0": self.pitch_head(condition).squeeze(-1),
            "condition": condition,
            "timbre": timbre,
            "content": content,
        }

    def distillation_prediction(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _, timbre, content = self.condition(batch)
        return self.distillation_head(timbre + content.mean(dim=1))

    @torch.no_grad()
    def sample(self, batch: dict[str, torch.Tensor], steps: int = 8) -> torch.Tensor:
        shape = (batch["phoneme_ids"].shape[0], batch["phoneme_ids"].shape[1], self.latent_dim)
        latent = torch.randn(shape, device=batch["phoneme_ids"].device)
        for index in range(steps):
            time = torch.full((shape[0],), index / steps, device=latent.device)
            output = self.forward(latent, time, batch)
            latent = latent + (output["velocity"] + 0.10 * output["acoustic_bias"]) / steps
        return latent


def grad_norm(module: nn.Module) -> float:
    return sum(float(parameter.grad.detach().abs().sum()) for parameter in module.parameters() if parameter.grad is not None)
