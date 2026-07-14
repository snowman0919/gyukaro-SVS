"""Whole-phrase ACE-Step + SoulX neural score renderer."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import atexit
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from gyu_singer.data import acoustic_reference_features
from gyu_singer.score import normalize_score

from .quality_controller import QualityPitchController


_STYLE = {"neutral": "neutral", "soft": "soft gentle", "breathy": "breathy", "energetic": "energetic", "dark": "dark emotional", "bright": "bright", "tense": "tense", "vibrato": "gentle vibrato"}
RESULT = "__GYU_RESULT__"
ERROR = "__GYU_ERROR__"


class _Worker:
    """One pinned-runtime model process, reused by all resident requests."""
    def __init__(self, command: list[str], cwd: Path, environment: dict[str, str]):
        self.process = subprocess.Popen(command, cwd=cwd, env=environment, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1)

    def request(self, body: dict) -> None:
        if not self.process.stdin or not self.process.stdout:
            raise RuntimeError("quality worker pipes unavailable")
        self.process.stdin.write(json.dumps(body) + "\n"); self.process.stdin.flush()
        for line in self.process.stdout:
            if line.startswith(RESULT): return
            if line.startswith(ERROR): raise RuntimeError(line.removeprefix(ERROR).strip())
        raise RuntimeError("quality worker exited before response")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill()


class SoulXPhraseRenderer:
    """Whole phrase neural content generation and neural timbre transfer."""
    sample_rate = 48000

    def __init__(self, reference: str | Path, root: str | Path = ".", use_controller: bool = True):
        self.reference, self.root = Path(reference), Path(root)
        self.cache = Path(os.environ.get("GYU_SINGER_CACHE", self.root / "data/cache"))
        self.ace_python = self.cache / "ace-step/.venv/bin/python"
        self.soulx_python = Path(os.environ.get("GYU_SOULX_PYTHON", self.root / ".venv-soulx/bin/python"))
        for executable in (self.ace_python, self.soulx_python):
            if not executable.exists(): raise FileNotFoundError(f"missing pinned neural runtime: {executable}")
        self.pitch_controller = None
        if use_controller:
            controller_path = self.root / "model/gyu_quality_pitch_controller.pt"
            if not controller_path.exists(): controller_path = self.root / "checkpoints/gyu_quality_pitch_controller.pt"
            if not controller_path.exists(): raise FileNotFoundError("missing quality pitch-controller checkpoint")
            self.pitch_controller = QualityPitchController(controller_path, acoustic_reference_features(self.reference))
        ace_environment = os.environ | {"PYTHONPATH": str(self.cache / "ace-step"), "GYU_SINGER_CACHE": str(self.cache)}
        soulx = self.cache / "soulx-singer"
        self.ace = _Worker([str(self.ace_python), "scripts/generate_ace_phrase.py", "--worker", "--checkpoint", str(self.cache / "ace-step-checkpoint")], self.root, ace_environment)
        self.soulx = _Worker([str(self.soulx_python), "scripts/probe_soulx_score.py", "--worker", "--reference", str(self.reference), "--model", str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"), "--config", str(soulx / "soulxsinger/config/soulxsinger.yaml"), "--rmvpe", str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")], self.root, os.environ | {"GYU_SINGER_CACHE": str(self.cache)})
        atexit.register(self.close)

    def model_info(self) -> dict:
        return {"backend": "hybrid-svs", "model_version": "gyu-hybrid-v0.3-quality", "checkpoint": "TriSinger pitch controller + ACE-Step-v1-3.5B + SoulX-Singer SVC", "languages": ["ko", "en", "ja"], "sample_rate": self.sample_rate, "resident_workers": True}

    @staticmethod
    def _f0(score: dict, duration: float, expressive: np.ndarray | None = None) -> np.ndarray:
        frames = max(1, round(duration * 50)); nominal = max(note["start"] + note["duration"] for note in score["notes"])
        times = np.arange(frames, dtype=np.float32) / 50 * nominal / duration
        points = score["curves"]["pitch"]
        residual = np.interp(times, [point["time"] for point in points], [point["value"] for point in points]) if points else np.zeros(frames, dtype=np.float32)
        if expressive is not None:
            residual = residual + np.interp(np.linspace(0, len(expressive) - 1, frames), np.arange(len(expressive)), expressive)
        values = np.zeros(frames, dtype=np.float32)
        for note in score["notes"]:
            active = (times >= note["start"]) & (times < note["start"] + note["duration"])
            values[active] = 440 * 2 ** ((note["pitch"] + (residual[active] if isinstance(residual, np.ndarray) else residual) - 69) / 12)
        return values

    def render(self, score: dict) -> np.ndarray:
        score = normalize_score(score); duration = max(note["start"] + note["duration"] for note in score["notes"])
        expressive = self.pitch_controller.predict(score)[0] if self.pitch_controller else None
        with tempfile.TemporaryDirectory(prefix="gyu-soulx-") as directory:
            temp = Path(directory); content, contour, output = temp / "content.wav", temp / "f0.npy", temp / "output.wav"
            self.ace.request({"language": score["language"], "lyrics": "\n".join(note["lyric"] for note in score["notes"]), "duration": duration, "style": _STYLE[score["style"]["preset"]], "output": str(content)})
            info = sf.info(content); np.save(contour, self._f0(score, info.frames / info.samplerate, None if expressive is None else expressive.cpu().numpy()))
            self.soulx.request({"source": str(content), "f0_npy": str(contour), "output": str(output)})
            audio, rate = sf.read(output, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        return resample_poly(mono, self.sample_rate, rate).astype(np.float32) if rate != self.sample_rate else mono

    def render_file(self, input_path: str | Path, output_path: str | Path) -> None:
        sf.write(output_path, self.render(json.loads(Path(input_path).read_text())), self.sample_rate, subtype="PCM_24")

    def close(self) -> None:
        for worker in (getattr(self, "ace", None), getattr(self, "soulx", None)):
            if worker: worker.close()
