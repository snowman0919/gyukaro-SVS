"""RC5 engineering path: canonical voicing/timing and safer SoulX decode."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .soulx import _Worker
from .content_timing import ctc_phone_alignment, latent_content_hold, latent_content_warp
from .v08 import GyuSingerV08Renderer


def soften_large_jumps(f0: np.ndarray, score: dict, duration_seconds: float = .08) -> np.ndarray:
    """Ramp the first voiced frames after octave-scale jumps; PITD bypasses this."""
    output = f0.copy()
    phrase_seconds = max(note["start"] + note["duration"] for note in score["notes"])
    frame_rate = len(f0) / phrase_seconds
    transition_frames = max(2, round(duration_seconds * frame_rate))
    for previous, note in zip(score["notes"], score["notes"][1:]):
        if abs(note["pitch"] - previous["pitch"]) < 12:
            continue
        onset = round(note["start"] * frame_rate)
        before = np.flatnonzero(f0[:onset] > 0)
        after = np.flatnonzero(f0[onset:] > 0)[:transition_frames] + onset
        if not len(before) or len(after) < 2:
            continue
        start = float(f0[before[-1]])
        alpha = np.arange(1, len(after) + 1, dtype="float32") / len(after)
        output[after] = np.exp((1 - alpha) * np.log(start) + alpha * np.log(f0[after]))
    return output


class GyuSingerV09Renderer(GyuSingerV08Renderer):
    """Post-RC4 canonical-timing and quality-decode path."""
    def __init__(self, reference: str | Path, root: str | Path = "."):
        super().__init__(reference, root=root)
        self.soulx.close()
        soulx = self.cache / "soulx-singer"
        command = [
            str(self.soulx_python), "scripts/probe_soulx_score.py", "--worker", "--precision", "fp32",
            "--reference", str(self.reference),
            "--model", str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"),
            "--config", str(soulx / "soulxsinger/config/soulxsinger.yaml"),
            "--rmvpe", str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
            "--latent-adapter", str(self.root / "checkpoints/gyu_real_latent_adapters_v0.7.pt"),
        ]
        self.soulx = _Worker(command, self.root, os.environ | {"GYU_SINGER_CACHE": str(self.cache)})
        self._ctc = None

    def _predict_pitch(self, score: dict) -> torch.Tensor:
        return self.pitch_controller.predict(score, canonical_timing=True)[0]

    def _target_f0(self, score: dict, duration: float, expressive: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        f0, timeline = self._canonical_f0(score, duration, expressive)
        if self._large_interval(score) and not score.get("curves", {}).get("pitch"):
            f0 = soften_large_jumps(f0, score)
            timeline = [row | {"f0_hz": float(f0[index])} for index, row in enumerate(timeline)]
        return f0, timeline

    @staticmethod
    def _rapid(score: dict) -> bool:
        return float(np.median([note["duration"] for note in score["notes"]])) <= .3

    @staticmethod
    def _large_interval(score: dict) -> bool:
        pitches = [note["pitch"] for note in score["notes"]]
        return any(abs(after - before) >= 12 for before, after in zip(pitches, pitches[1:]))

    def _decoder_options(self, score: dict | None = None) -> dict:
        if score and self._large_interval(score):
            return {"n_steps": 50, "cfg": 2.0, "seed": 21}
        if score and not self._rapid(score) and score["language"] in {"ko", "en"} and score["style"]["preset"] == "neutral" and all(note["duration"] < 4 for note in score["notes"]) and all(after["start"] <= before["start"] + before["duration"] + .05 for before, after in zip(score["notes"], score["notes"][1:])):
            return {"n_steps": 32, "cfg": 1.5, "seed": 21}
        return {"n_steps": 64, "cfg": 2.0, "seed": 21}

    @classmethod
    def _content_warp_strength(cls, score: dict) -> float:
        if cls._rapid(score):
            return 1.0
        if score["language"] == "en":
            return .25
        contiguous = all(
            after["start"] <= before["start"] + before["duration"] + .05
            for before, after in zip(score["notes"], score["notes"][1:])
        )
        if (
            score["language"] == "ko"
            and score["style"]["preset"] == "neutral"
            and not cls._large_interval(score)
            and contiguous
        ):
            return .05
        return 0.0

    def _content_options(self, score: dict, content: Path, target_f0: np.ndarray, temp: Path) -> dict:
        strength = self._content_warp_strength(score)
        if not strength:
            return {}
        import torchaudio
        if self._ctc is None:
            bundle = torchaudio.pipelines.MMS_FA
            self._ctc = (bundle.get_model().eval(), bundle.get_labels())
        audio, rate = sf.read(content, dtype="float32", always_2d=True)
        mono = audio.mean(1)
        aligned = resample_poly(mono, 16_000, rate).astype("float32") if rate != 16_000 else mono
        alignment = ctc_phone_alignment(torch.from_numpy(aligned), 16_000, score, *self._ctc)
        duration = len(mono) / rate
        warp = latent_content_hold(alignment, duration, len(target_f0)) if self._rapid(score) else latent_content_warp(alignment, duration, len(target_f0) / 50, len(target_f0))
        path = temp / "content_warp.npy"; np.save(path, warp)
        return {"content_warp_npy": str(path), "content_warp_strength": strength}

    def render(self, score: dict) -> np.ndarray:
        audio = np.clip(super().render(score), -1, 1)
        peak = float(np.max(np.abs(audio)))
        return audio * min(1.0, .97 / max(peak, 1e-8))

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-v0.9-timing-experiment",
            "model_version": "gyu-singer-v0.9-timing-experiment",
            "rc4_preserved_backend": "gyu-singer-v0.8",
            "canonical_phone_score_timeline": True,
            "unvoiced_f0_zero": True,
            "soulx_decode_policy": "measured policy: 32/CFG1.5 standard, 64/CFG2 rapid, 50/CFG2 large interval",
            "soulx_precision": "fp32",
            "content_timing": "MMS CTC latent hold for rapid phrases; 0.25 English and measured 0.05 normal-KO latent warp; large-interval content unchanged",
            "large_interval_transition": "80 ms score-domain onset ramp when PITD is absent",
            "human_listening_status": "candidate4_passed_2026-07-15",
        }
