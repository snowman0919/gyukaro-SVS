"""v0.6 phrase renderer: v0.5 prosody + shared identity + SoulX latent FiLM."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gyu_singer.data import acoustic_reference_features
from gyu_singer.model import MultiTeacherIdentityEncoder
from gyu_singer.score import normalize_score

from .acoustic_style import adapt_waveform, load_adapter
from .v05 import GyuSingerV05Renderer


class GyuSingerV06Renderer(GyuSingerV05Renderer):
    def __init__(self, reference: str | Path, root: str | Path = ".", latent_adapter_enabled: bool = True, latent_adapter_checkpoint: str | Path | None = None):
        self.root = Path(root)
        self.latent_adapter_enabled = latent_adapter_enabled
        checkpoint = latent_adapter_checkpoint or "checkpoints/gyu_latent_adapter_v0.6.pt"
        super().__init__(reference, root=root, latent_adapter_checkpoint=checkpoint if latent_adapter_enabled else None)
        # Independent-score evaluation did not show a consistent v0.6 prosody gain, so the
        # production v0.6 path retains the measured v0.5 controller.  The v0.6 checkpoint is
        # packaged only as an explicit experimental baseline.
        path = self.root / "checkpoints/gyu_identity_space_v0.6.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.identity_encoder = MultiTeacherIdentityEncoder(**saved["model_config"]).to(self.pitch_controller.device).eval()
        self.identity_encoder.load_state_dict(saved["model"])
        self.identity_enabled = True
        self.style_enabled = True
        self.identity_mode = "student"
        self.teacher_fish = self.teacher_moss = None
        fish_path = self.root / "data/cache/teacher_representations/fish/teacher_ko_001.pt"
        moss_path = self.root / "data/cache/teacher_representations/moss/teacher_ko_001.pt"
        if fish_path.exists() and moss_path.exists():
            self.teacher_fish = torch.load(fish_path, weights_only=True).float().flatten().to(self.pitch_controller.device)
            self.teacher_moss = torch.load(moss_path, weights_only=True).float().flatten().to(self.pitch_controller.device)

    def model_info(self) -> dict:
        return {"backend": "gyu-singer-v0.6", "model_version": "gyu-singer-v0.6-experimental", "prosody_checkpoint": "checkpoints/gyu_prosody_v0.5.pt", "prosody_v0_6_checkpoint": "checkpoints/gyu_prosody_v0.6.pt (experimental baseline; not selected)", "identity_checkpoint": "checkpoints/gyu_identity_space_v0.6.pt", "latent_adapter_checkpoint": "checkpoints/gyu_latent_adapter_v0.6.pt", "content": "OmniVoice", "decoder": "SoulX-Singer SVC", "languages": ["ko", "en", "ja"], "sample_rate": self.sample_rate, "v0_4_fallback": False, "per_note_tts": False, "waveform_pitch_shift": False}

    @staticmethod
    def _style_vector(style: dict, device: torch.device) -> torch.Tensor:
        vector = torch.zeros(64, device=device)
        preset = {"neutral": 0, "soft": 1, "breathy": 2, "energetic": 3, "dark": 4, "bright": 5}.get(style["preset"], 0)
        vector[preset] = 1.0
        controls = [style.get(name, 0.0) for name in ("dynamics", "breathiness", "tension", "brightness", "vibrato")]
        vector[8:13] = torch.tensor(controls, device=device)
        return vector

    def _content_style_preset(self, style: dict) -> str:
        """Hook kept identical for v0.6; v0.7 isolates latent style causality."""
        return style["preset"]

    def _identity_vector(self) -> torch.Tensor:
        if not self.identity_enabled or self.identity_mode == "none":
            return torch.zeros(64, device=self.pitch_controller.device)
        if self.identity_mode == "student":
            return self.identity_encoder.student(self.reference_features[None])[0]
        if self.teacher_fish is None or self.teacher_moss is None:
            raise RuntimeError("teacher ablation requires packaged Fish/MOSS representations")
        fish = torch.nn.functional.normalize(self.identity_encoder.fish_projection(self.teacher_fish[None]), dim=-1)[0]
        if self.identity_mode == "fish_only":
            return fish
        moss = torch.nn.functional.normalize(self.identity_encoder.moss_projection(self.teacher_moss[None]), dim=-1)[0]
        if self.identity_mode == "fish_moss":
            return torch.nn.functional.normalize(fish + moss, dim=-1)
        raise ValueError(f"unknown identity mode: {self.identity_mode}")

    def _predict_pitch(self, score: dict) -> torch.Tensor:
        return self.pitch_controller.predict(score)[0]

    def _target_f0(self, score: dict, duration: float, expressive: np.ndarray) -> tuple[np.ndarray, list[dict] | None]:
        return self._f0(score, duration, expressive), None

    def _decoder_options(self, score: dict | None = None) -> dict:
        return {}

    def _content_options(self, score: dict, content: Path, target_f0: np.ndarray, temp: Path) -> dict:
        return {}

    def _generate_content(self, score: dict, duration: float, content: Path, temp: Path) -> None:
        self.omnivoice.request({
            "language": score["language"],
            "lyrics": "".join(note["lyric"] for note in score["notes"]),
            "duration": duration,
            "output": str(content),
        })

    def render(self, score: dict) -> np.ndarray:
        score = normalize_score(score); duration = max(note["start"] + note["duration"] for note in score["notes"])
        strength = float(score["style"]["prosody_strength"])
        expressive = self._predict_pitch(score) * strength
        from .quality_controller import STYLE
        style = score["style"]; preset = torch.tensor(STYLE[self._content_style_preset(style)], device=self.pitch_controller.device)
        identity = self._identity_vector()
        style_vector = self._style_vector(style, self.pitch_controller.device) if self.style_enabled else torch.zeros(64, device=self.pitch_controller.device)
        identity_ref = self.reference_features + .05 * identity.repeat((self.reference_features.shape[0] + identity.shape[0] - 1) // identity.shape[0])[: self.reference_features.shape[0]]
        controls = np.array([.8, 0, 0, 0, 0], dtype="float32")
        points = score["curves"]
        for index, name in enumerate(("dynamics", "breathiness", "tension", "brightness", "vibrato")):
            if points[name]: controls[index] = float(np.mean([point["value"] for point in points[name]]))
        with tempfile.TemporaryDirectory(prefix="gyu-soulx-v06-") as directory:
            temp = Path(directory); content, contour, output = temp / "content.wav", temp / "f0.npy", temp / "output.wav"
            identity_path, style_path = temp / "identity.npy", temp / "style.npy"
            np.save(identity_path, identity.detach().cpu().numpy()); np.save(style_path, style_vector.detach().cpu().numpy())
            self._generate_content(score, duration, content, temp)
            content_audio, content_rate = sf.read(content, dtype="float32", always_2d=True); content_audio = content_audio.mean(1)
            content_audio = adapt_waveform(content_audio, content_rate, self.acoustic_adapter, identity_ref, torch.from_numpy(controls).to(self.pitch_controller.device), preset, style["acoustic_style_strength"])
            sf.write(content, content_audio, content_rate, subtype="PCM_16")
            info = sf.info(content); target_f0, _ = self._target_f0(score, info.frames / info.samplerate, expressive.cpu().numpy()); np.save(contour, target_f0)
            self.soulx.request({"source": str(content), "f0_npy": str(contour), "output": str(output), "identity_npy": str(identity_path) if self.identity_enabled and self.identity_mode != "none" else None, "style_npy": str(style_path) if self.style_enabled else None} | self._decoder_options(score) | self._content_options(score, content, target_f0, temp))
            audio, rate = sf.read(output, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        from scipy.signal import resample_poly
        return resample_poly(mono, self.sample_rate, rate).astype(np.float32) if rate != self.sample_rate else mono
