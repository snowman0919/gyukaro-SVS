"""Overlap-safe runtime for the measured spectral refiner checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gyu_singer.model import SpectralAcousticRefiner


class SpectralRefinerRuntime:
    def __init__(self, path: str | Path, device: str | None = None):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.mode = saved["stage"]
        self.model = SpectralAcousticRefiner(**saved["model_config"])
        self.model.load_state_dict(saved["model"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def process(self, audio: np.ndarray, chunk: int = 192_000, overlap: int = 4096) -> np.ndarray:
        source = np.asarray(audio, dtype="float32")
        if len(source) <= chunk:
            with torch.inference_mode():
                return self.model(torch.from_numpy(source)[None].to(self.device), self.mode)[0].cpu().numpy()
        step = chunk - overlap
        output, weights = np.zeros_like(source), np.zeros_like(source)
        for start in range(0, len(source), step):
            end = min(len(source), start + chunk)
            with torch.inference_mode():
                value = self.model(torch.from_numpy(source[start:end])[None].to(self.device), self.mode)[0].cpu().numpy()
            window = np.ones(end - start, dtype="float32")
            fade = min(overlap, len(window) // 2)
            if start:
                window[:fade] = np.linspace(0, 1, fade, dtype="float32")
            if end < len(source):
                window[-fade:] = np.linspace(1, 0, fade, dtype="float32")
            output[start:end] += value * window; weights[start:end] += window
            if end == len(source):
                break
        return output / np.maximum(weights, 1e-6)
