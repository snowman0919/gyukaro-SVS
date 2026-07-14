"""Whole-phrase ACE-Step + SoulX neural score renderer."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from gyu_singer.score import normalize_score


_STYLE = {"neutral": "neutral", "soft": "soft gentle", "breathy": "breathy", "energetic": "energetic", "dark": "dark emotional", "bright": "bright", "tense": "tense", "vibrato": "gentle vibrato"}


class SoulXPhraseRenderer:
    """Whole phrase neural content generation and neural timbre transfer."""
    sample_rate = 48000

    def __init__(self, reference: str | Path, root: str | Path = "."):
        self.reference, self.root = Path(reference), Path(root)
        self.cache = Path(os.environ.get("GYU_SINGER_CACHE", self.root / "data/cache"))
        self.ace_python = self.cache / "ace-step/.venv/bin/python"
        self.soulx_python = Path(os.environ.get("GYU_SOULX_PYTHON", self.root / ".venv-soulx/bin/python"))
        for executable in (self.ace_python, self.soulx_python):
            if not executable.exists(): raise FileNotFoundError(f"missing pinned neural runtime: {executable}")

    def model_info(self) -> dict:
        return {"backend": "hybrid-soulx-phrase", "model_version": "gyu-hybrid-v0.3-quality-probe", "checkpoint": "ACE-Step-v1-3.5B + SoulX-Singer SVC (frozen)", "languages": ["ko", "en", "ja"], "sample_rate": self.sample_rate}

    @staticmethod
    def _f0(score: dict, duration: float) -> np.ndarray:
        frames = max(1, round(duration * 50)); nominal = max(note["start"] + note["duration"] for note in score["notes"])
        times = np.arange(frames, dtype=np.float32) / 50 * nominal / duration
        points = score["curves"]["pitch"]
        residual = np.interp(times, [point["time"] for point in points], [point["value"] for point in points]) if points else 0.0
        values = np.zeros(frames, dtype=np.float32)
        for note in score["notes"]:
            active = (times >= note["start"]) & (times < note["start"] + note["duration"])
            values[active] = 440 * 2 ** ((note["pitch"] + (residual[active] if isinstance(residual, np.ndarray) else residual) - 69) / 12)
        return values

    def render(self, score: dict) -> np.ndarray:
        score = normalize_score(score); duration = max(note["start"] + note["duration"] for note in score["notes"])
        with tempfile.TemporaryDirectory(prefix="gyu-soulx-") as directory:
            temp = Path(directory); content, contour, output = temp / "content.wav", temp / "f0.npy", temp / "output.wav"
            environment = os.environ | {"PYTHONPATH": str(self.cache / "ace-step")}
            subprocess.run([str(self.ace_python), "scripts/generate_ace_phrase.py", "--checkpoint", str(self.cache / "ace-step-checkpoint"), "--language", score["language"], "--lyrics", "\n".join(note["lyric"] for note in score["notes"]), "--duration", str(duration), "--style", _STYLE[score["style"]["preset"]], "--output", str(content)], cwd=self.root, env=environment, check=True)
            info = sf.info(content); np.save(contour, self._f0(score, info.frames / info.samplerate))
            soulx = self.cache / "soulx-singer"
            subprocess.run([str(self.soulx_python), "scripts/probe_soulx_score.py", "--source", str(content), "--reference", str(self.reference), "--f0-npy", str(contour), "--model", str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"), "--config", str(soulx / "soulxsinger/config/soulxsinger.yaml"), "--rmvpe", str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), "--output", str(output)], cwd=self.root, check=True)
            audio, rate = sf.read(output, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        return resample_poly(mono, self.sample_rate, rate).astype(np.float32) if rate != self.sample_rate else mono

    def render_file(self, input_path: str | Path, output_path: str | Path) -> None:
        sf.write(output_path, self.render(json.loads(Path(input_path).read_text())), self.sample_rate, subtype="PCM_24")
