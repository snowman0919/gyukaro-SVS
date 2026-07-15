"""Measured post-SoulX refiner runtime with overlap-safe long-form inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gyu_singer.model import VocalAcousticRefiner


class AcousticRefinerRuntime:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.mode = saved["stage"]
        self.model = VocalAcousticRefiner(**saved["model_config"])
        self.model.load_state_dict(saved["model"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def process(self, audio: np.ndarray, chunk_samples: int = 192_000, overlap: int = 8_192) -> np.ndarray:
        source = np.asarray(audio, dtype="float32")
        if len(source) <= chunk_samples:
            with torch.inference_mode():
                return self.model(torch.from_numpy(source)[None].to(self.device), self.mode)[0].cpu().numpy()
        step = chunk_samples - overlap
        output, weights = np.zeros_like(source), np.zeros_like(source)
        for start in range(0, len(source), step):
            end = min(len(source), start + chunk_samples)
            with torch.inference_mode():
                refined = self.model(torch.from_numpy(source[start:end])[None].to(self.device), self.mode)[0].cpu().numpy()
            window = np.ones(end - start, dtype="float32")
            fade = min(overlap, len(window) // 2)
            if start > 0:
                window[:fade] = np.linspace(0, 1, fade, dtype="float32")
            if end < len(source):
                window[-fade:] = np.linspace(1, 0, fade, dtype="float32")
            output[start:end] += refined * window; weights[start:end] += window
            if end == len(source):
                break
        return output / np.maximum(weights, 1e-6)
