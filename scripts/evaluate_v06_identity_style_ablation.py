#!/usr/bin/env python3
"""Measure whether v0.6 identity/style conditioning changes final SoulX audio."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from gyu_singer.inference.v06 import GyuSingerV06Renderer

def audio16(path):
    a, r = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    return resample_poly(a, 16000, r).astype("float32") if r != 16000 else a

def main():
    score = json.loads(Path("examples/quality_ko.json").read_text())
    renderer = GyuSingerV06Renderer("data/processed/master/216.wav")
    variants = {"identity_enabled": (True, True), "identity_zero": (False, True), "latent_style_zero": (True, False)}
    paths = {}
    try:
        for label, (identity, style) in variants.items():
            renderer.identity_enabled, renderer.style_enabled = identity, style
            path = Path("artifacts/reports") / f"v06_ablation_{label}.wav"
            sf.write(path, renderer.render(score), renderer.sample_rate, subtype="PCM_16"); paths[label] = str(path)
    finally:
        renderer.close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    def emb(path):
        inp = extractor(audio16(path), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): value = wavlm(**{k: v.to(device) for k, v in inp.items()}).embeddings
        return torch.nn.functional.normalize(value, dim=-1).squeeze().cpu().numpy()
    ref = emb("data/processed/master/216.wav"); base = emb(paths["identity_enabled"])
    metrics = {"wavlm_to_gyu": {}, "audio_rms": {}, "pairwise_l2": {}}
    vectors = {}
    for label, path in paths.items():
        vectors[label] = emb(path); audio = audio16(path); metrics["wavlm_to_gyu"][label] = round(float(np.dot(ref, vectors[label])), 5); metrics["audio_rms"][label] = round(float(np.sqrt(np.mean(audio * audio))), 6)
    for label in paths:
        metrics["pairwise_l2"][label] = round(float(np.sqrt(np.mean((vectors[label] - base) ** 2))), 6)
    report = {"paths": paths, "metric": "WavLM speaker cosine; same KO score/content/F0", "identity_effect": metrics["pairwise_l2"]["identity_zero"] > 1e-5, "latent_style_effect": metrics["pairwise_l2"]["latent_style_zero"] > 1e-5, "metrics": metrics, "ecapa": "not run: v0.6 gate uses WavLM; add ECAPA when pinned evaluator is available"}
    Path("artifacts/reports/v06_identity_style_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/soulx_identity_adapter.md").write_text("# v0.6 SoulX identity adapter\n\n" + json.dumps(report, indent=2) + "\n")
    Path("docs/latent_style_adapter.md").write_text("# v0.6 latent acoustic-style adapter\n\n" + json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
