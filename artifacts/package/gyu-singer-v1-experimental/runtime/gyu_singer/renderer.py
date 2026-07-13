from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample


def midi_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


class Renderer:
    """Resident, score-controlled renderer backed by adapted GYU voice loops."""

    def __init__(self, model_path: str | Path):
        bank = np.load(model_path, allow_pickle=False)
        self.loops = bank["loops"].astype(np.float32)
        self.f0 = bank["f0"].astype(np.float32)
        self.sample_rate = int(bank["sample_rate"])

    def render(self, score: dict) -> np.ndarray:
        rate = int(score.get("sample_rate", self.sample_rate))
        notes = score.get("notes", [])
        if not notes:
            raise ValueError("score.notes must not be empty")
        end = max(float(n["start"]) + float(n["duration"]) for n in notes)
        output = np.zeros(int((end + 0.08) * rate), np.float32)
        for note in notes:
            start, duration = float(note["start"]), float(note["duration"])
            frames = max(1, int(duration * rate))
            target = midi_hz(float(note["pitch"]))
            choice = int(np.argmin(abs(self.f0 - target)))
            loop = self.loops[choice]
            # Source resampling gives explicit pitch control; loop preserves GYU timbre.
            source_frames = max(8, int(len(loop) * self.f0[choice] / target))
            pitched = resample(loop, source_frames).astype(np.float32)
            repeated = np.resize(pitched, frames)
            fade = min(int(rate * 0.025), frames // 3)
            env = np.ones(frames, np.float32)
            if fade:
                ramp = np.linspace(0, 1, fade, dtype=np.float32)
                env[:fade], env[-fade:] = ramp, ramp[::-1]
            dynamics = float(note.get("dynamics", 0.8))
            offset = int(start * rate)
            output[offset : offset + frames] += repeated * env * dynamics
        peak = float(np.max(np.abs(output)))
        return output / max(1.0, peak / 0.92)

    def render_file(self, input_path: str | Path, output_path: str | Path) -> None:
        score = json.loads(Path(input_path).read_text())
        audio = self.render(score)
        sf.write(output_path, audio, int(score.get("sample_rate", self.sample_rate)), subtype="PCM_24")
