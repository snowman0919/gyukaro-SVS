#!/usr/bin/env python3
"""Measure GYU speaker similarity on local full-song phrase segments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
WORK = ROOT / "data/external/work/rc9_reference"


def mono16(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    return resample_poly(audio, 16_000, rate).astype("float32") if rate != 16_000 else audio


def main() -> None:
    render = mono16(WORK / "openutau_render.wav")
    reference = mono16(ROOT / "data/processed/master/216.wav")
    project = yaml.safe_load((WORK / "nonbreath_oblige_gyu_rc9.ustx").read_text())
    requests = json.loads((WORK / "openutau_phrase_requests.json").read_text())
    extractor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"), savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )

    def embedding(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = extractor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            first = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            second = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        first = torch.nn.functional.normalize(first, dim=-1).squeeze().cpu().numpy()
        second = second.squeeze().cpu().numpy(); second /= max(np.linalg.norm(second), 1e-8)
        return first, second

    reference_embedding = embedding(reference)
    bpm = float(project["tempos"][0]["bpm"])
    rows = []
    for index, (part, request) in enumerate(zip(project["voice_parts"], requests), 1):
        start = round(float(part["position"]) / 480 * 60 / bpm * 16_000)
        duration = max(float(note["start"]) + float(note["duration"]) for note in request["notes"])
        segment = render[start:min(len(render), start + round(duration * 16_000))]
        current = embedding(segment)
        rows.append({
            "phrase": index,
            "language": request["language"],
            "wavlm_to_gyu": round(float(np.dot(reference_embedding[0], current[0])), 5),
            "ecapa_to_gyu": round(float(np.dot(reference_embedding[1], current[1])), 5),
        })
    wavlm_values = np.array([row["wavlm_to_gyu"] for row in rows])
    ecapa_values = np.array([row["ecapa_to_gyu"] for row in rows])
    summary = {
        "phrases": len(rows),
        "wavlm_mean": round(float(wavlm_values.mean()), 5),
        "wavlm_median": round(float(np.median(wavlm_values)), 5),
        "wavlm_p10": round(float(np.percentile(wavlm_values, 10)), 5),
        "ecapa_mean": round(float(ecapa_values.mean()), 5),
        "ecapa_median": round(float(np.median(ecapa_values)), 5),
    }
    report = {
        "status": "identity_nonregression_pass_human_pending" if summary["wavlm_median"] >= .58 else "identity_gate_fail",
        "reference": "data/processed/master/216.wav",
        "summary": summary,
        "phrases": rows,
        "human_listening_required": True,
        "copyright": "local song render excluded from Git and package; report contains metrics only",
    }
    output = ROOT / "artifacts/reports/reference_song_rc9_identity.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": summary}, indent=2))
    if report["status"] == "identity_gate_fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
