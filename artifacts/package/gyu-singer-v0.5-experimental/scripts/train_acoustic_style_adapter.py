#!/usr/bin/env python3
"""Fit compact spectral adapter on real GYU audio plus low-trust teacher style rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft

from gyu_singer.data import acoustic_reference_features, read_jsonl
from gyu_singer.inference.acoustic_style import GyuAcousticStyleAdapter

STYLE = {"neutral": 0, "soft": 1, "breathy": 2, "energetic": 3, "dark": 4, "bright": 5, "tense": 6, "vibrato": 7}


def target(path: str, bins: int = 257) -> torch.Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
    _, _, z = stft(audio, fs=rate, nperseg=512, noverlap=384, boundary="zeros")
    value = np.log(np.maximum(np.abs(z), 1e-5)).mean(1)
    return torch.from_numpy(np.interp(np.linspace(0, len(value) - 1, bins), np.arange(len(value)), value).astype("float32"))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(19)
    real = read_jsonl("data/manifests/neural_supervision.jsonl")
    teachers = read_jsonl("data/manifests/teacher_style_supplement_weighted.jsonl")
    rows = [(r["audio_path"], "neutral", 1.0) for r in real] + [(r["output_path"], r.get("style", "neutral"), float(r["trust_weight"])) for r in teachers]
    adapter = GyuAcousticStyleAdapter().to(device).train(); opt = torch.optim.AdamW(adapter.parameters(), lr=2e-3); history = []
    for step in range(1, 401):
        path, style, weight = rows[(step - 1) % len(rows)]; ref = acoustic_reference_features(path, strict_sample_rate=False).to(device); controls = torch.tensor([[.8, 0, 0, 0, .7 if style == "vibrato" else 0]], device=device); preset = torch.tensor([STYLE.get(style, 0)], device=device)
        pred = adapter(ref[None], controls, preset)[0]; truth = target(path).to(device); loss = ((pred - truth / 8) ** 2).mean() * weight
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0: history.append({"step": step, "loss": round(float(loss.detach()), 6)})
    output = "checkpoints/gyu_acoustic_style_adapter_v0.5.pt"; Path(output).parent.mkdir(exist_ok=True); torch.save({"model": adapter.eval().cpu().state_dict(), "config": {"feature_dim": 160, "output_bins": 257}, "input": "GYU_reference_features_plus_style_controls", "target": "log_spectral_envelope", "real_rows": len(real), "teacher_rows": len(teachers)}, output)
    report = {"steps": 400, "real_rows": len(real), "teacher_rows": len(teachers), "teacher_role": "low_trust_style_structure_only", "history": history, "checkpoint": output}
    Path("artifacts/reports/acoustic_style_adapter_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__": main()
