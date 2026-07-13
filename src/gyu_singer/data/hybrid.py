from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def acoustic_reference_features(path: str | Path, sample_rate: int = 48000) -> torch.Tensor:
    """160-D fixed reference summary; used only for timbre conditioning."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.mean(axis=1))
    if rate != sample_rate:
        # Dataset masters are canonical 48 kHz; reject mismatches instead of hidden DSP conversion.
        raise ValueError(f"reference must be {sample_rate} Hz, got {rate}")
    if waveform.numel() < 1024:
        waveform = torch.nn.functional.pad(waveform, (0, 1024 - waveform.numel()))
    spectrum = torch.stft(waveform, n_fft=1024, hop_length=480, window=torch.hann_window(1024), return_complex=True).abs()
    bands = torch.nn.functional.interpolate(spectrum[None, None], size=(80, spectrum.shape[-1]), mode="bilinear", align_corners=False)[0, 0]
    return torch.cat((bands.mean(dim=1), bands.std(dim=1).clamp_min(1e-5))).float()
