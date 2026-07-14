"""Distinct v0.5 renderer: real-GYU prosody controller plus acoustic adapter."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gyu_singer.data import acoustic_reference_features
from gyu_singer.score import normalize_score

from .acoustic_style import adapt_waveform, load_adapter
from .soulx import SoulXPhraseRenderer
from gyu_singer.model import MultiTeacherIdentityEncoder


class GyuSingerV05Renderer(SoulXPhraseRenderer):
    def __init__(self, reference: str | Path, root: str | Path = ".", latent_adapter_checkpoint: str | Path | None = None):
        super().__init__(reference, root=root, use_controller=True, controller_checkpoint="checkpoints/gyu_prosody_v0.5.pt", latent_adapter_checkpoint=latent_adapter_checkpoint)
        checkpoint = self.root / "checkpoints/gyu_acoustic_style_adapter_v0.5.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing v0.5 acoustic-style adapter: {checkpoint}")
        self.acoustic_adapter = load_adapter(checkpoint, self.pitch_controller.device)
        self.reference_features = acoustic_reference_features(self.reference).to(self.pitch_controller.device)
        teacher_path = self.root / "checkpoints/gyu_teacher_identity_v0.5.pt"
        if not teacher_path.exists():
            raise FileNotFoundError(f"missing v0.5 teacher identity checkpoint: {teacher_path}")
        saved = torch.load(teacher_path, map_location="cpu", weights_only=False)
        self.teacher_identity = MultiTeacherIdentityEncoder(**saved["model_config"]).to(self.pitch_controller.device).eval()
        self.teacher_identity.load_state_dict(saved["model"])

    def model_info(self) -> dict:
        return {"backend": "gyu-singer-v0.5", "model_version": "gyu-singer-v0.5-experimental", "prosody_checkpoint": "checkpoints/gyu_prosody_v0.5.pt", "acoustic_style_checkpoint": "checkpoints/gyu_acoustic_style_adapter_v0.5.pt", "content": "OmniVoice", "decoder": "SoulX-Singer SVC", "languages": ["ko", "en", "ja"], "sample_rate": self.sample_rate, "v0_4_fallback": False}

    def render(self, score: dict) -> np.ndarray:
        score = normalize_score(score); duration = max(note["start"] + note["duration"] for note in score["notes"])
        strength = float(score["style"]["prosody_strength"])
        expressive = self.pitch_controller.predict(score)[0] * strength
        from .quality_controller import STYLE
        style = score["style"]; preset = torch.tensor(STYLE[style["preset"]], device=self.pitch_controller.device)
        identity = self.teacher_identity.student(self.reference_features[None])[0]
        identity_ref = self.reference_features + .05 * identity.repeat((self.reference_features.shape[0] + identity.shape[0] - 1) // identity.shape[0])[: self.reference_features.shape[0]]
        controls = np.array([.8, 0, 0, 0, 0], dtype="float32")
        points = score["curves"]
        for index, name in enumerate(("dynamics", "breathiness", "tension", "brightness", "vibrato")):
            if points[name]: controls[index] = float(np.mean([point["value"] for point in points[name]]))
        style_controls = torch.from_numpy(controls).to(self.pitch_controller.device)
        with tempfile.TemporaryDirectory(prefix="gyu-soulx-v05-") as directory:
            temp = Path(directory); content, contour, output = temp / "content.wav", temp / "f0.npy", temp / "output.wav"
            self.omnivoice.request({"language": score["language"], "lyrics": "".join(note["lyric"] for note in score["notes"]), "duration": duration, "output": str(content)})
            content_audio, content_rate = sf.read(content, dtype="float32", always_2d=True); content_audio = content_audio.mean(1)
            content_audio = adapt_waveform(content_audio, content_rate, self.acoustic_adapter, identity_ref, style_controls, preset, style["acoustic_style_strength"])
            sf.write(content, content_audio, content_rate, subtype="PCM_16")
            info = sf.info(content); np.save(contour, self._f0(score, info.frames / info.samplerate, expressive.cpu().numpy()))
            self.soulx.request({"source": str(content), "f0_npy": str(contour), "output": str(output)})
            audio, rate = sf.read(output, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        from scipy.signal import resample_poly
        return resample_poly(mono, self.sample_rate, rate).astype(np.float32) if rate != self.sample_rate else mono
