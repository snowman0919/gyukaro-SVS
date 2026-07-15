#!/usr/bin/env python3
"""Measure whether the acoustic refiner changes GYU speaker identity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector


def audio16(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); mono = audio.mean(1)
    return resample_poly(mono, 16000, rate).astype("float32") if rate != 16000 else mono


def normalized(value: np.ndarray) -> np.ndarray:
    return value / max(np.linalg.norm(value), 1e-8)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    ecapa = EncoderClassifier.from_hparams(source="data/cache/spkrec-ecapa-voxceleb", savedir="data/cache/spkrec-ecapa-voxceleb", run_opts={"device": device})
    def embed(path: str) -> tuple[np.ndarray, np.ndarray]:
        audio = audio16(path); values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            first = wavlm(**{key: value.to(device) for key, value in values.items()}).embeddings.squeeze().float().cpu().numpy()
            second = ecapa.encode_batch(torch.from_numpy(audio)[None].to(device)).squeeze().float().cpu().numpy()
        return normalized(first), normalized(second)
    reference = embed("data/processed/master/216.wav")
    before = json.loads(Path("artifacts/reports/rc5_stress_candidate4/manifest.json").read_text())
    after = json.loads(Path("artifacts/reports/refiner_rc_candidate/manifest.json").read_text())
    rows = []
    for case in before["files"]:
        baseline, candidate = embed(before["files"][case]["path"]), embed(after["files"][case]["path"])
        rows.append({"case": case,
                     "wavlm_before_to_gyu": round(float(np.dot(baseline[0], reference[0])), 6), "wavlm_after_to_gyu": round(float(np.dot(candidate[0], reference[0])), 6),
                     "ecapa_before_to_gyu": round(float(np.dot(baseline[1], reference[1])), 6), "ecapa_after_to_gyu": round(float(np.dot(candidate[1], reference[1])), 6),
                     "wavlm_before_after": round(float(np.dot(baseline[0], candidate[0])), 6), "ecapa_before_after": round(float(np.dot(baseline[1], candidate[1])), 6)})
    aggregate = {key: round(float(np.mean([row[key] for row in rows])), 6) for key in rows[0] if key != "case"}
    aggregate |= {"wavlm_to_gyu_delta": round(aggregate["wavlm_after_to_gyu"] - aggregate["wavlm_before_to_gyu"], 6),
                  "ecapa_to_gyu_delta": round(aggregate["ecapa_after_to_gyu"] - aggregate["ecapa_before_to_gyu"], 6)}
    report = {"status": "identity_preservation_diagnostic_not_listening_evidence", "candidate_strength": .25, "aggregate": aggregate, "rows": rows}
    Path("artifacts/reports/refiner_identity_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
