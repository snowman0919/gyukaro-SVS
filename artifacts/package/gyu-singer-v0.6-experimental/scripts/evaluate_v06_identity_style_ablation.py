#!/usr/bin/env python3
"""Multi-language Fish/MOSS identity and SoulX latent-style ablations."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

from gyu_singer.inference.v06 import GyuSingerV06Renderer


def audio16(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
    return resample_poly(audio, 16000, rate).astype("float32") if rate != 16000 else audio


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = GyuSingerV06Renderer("data/processed/master/216.wav")
    paths = {}
    try:
        for language in ("ko", "en", "ja"):
            score = json.loads(Path(f"examples/quality_{language}.json").read_text())
            for mode in ("none", "fish_only", "fish_moss", "student"):
                renderer.identity_mode, renderer.identity_enabled, renderer.style_enabled = mode, mode != "none", True
                path = Path("artifacts/reports") / f"v06_ablation_{mode}_{language}.wav"
                sf.write(path, renderer.render(score), renderer.sample_rate, subtype="PCM_16")
                paths[f"{language}:{mode}"] = str(path)
        score = json.loads(Path("examples/quality_ko.json").read_text())
        renderer.identity_mode, renderer.identity_enabled, renderer.style_enabled = "student", True, False
        path = Path("artifacts/reports/v06_ablation_latent_style_zero.wav")
        sf.write(path, renderer.render(score), renderer.sample_rate, subtype="PCM_16"); paths["ko:latent_style_zero"] = str(path)
    finally:
        renderer.close()

    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    ecapa = EncoderClassifier.from_hparams(source="data/cache/spkrec-ecapa-voxceleb", savedir="data/cache/spkrec-ecapa-voxceleb", run_opts={"device": device})

    def wavlm_emb(path):
        inputs = extractor(audio16(path), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): value = wavlm(**{key: item.to(device) for key, item in inputs.items()}).embeddings
        return torch.nn.functional.normalize(value, dim=-1).squeeze().cpu().numpy()

    def ecapa_emb(path):
        with torch.inference_mode(): value = ecapa.encode_batch(torch.from_numpy(audio16(path)).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        return value / max(np.linalg.norm(value), 1e-8)

    reference_wavlm, reference_ecapa = wavlm_emb("data/processed/master/216.wav"), ecapa_emb("data/processed/master/216.wav")
    metrics = {"wavlm_to_gyu": {}, "ecapa_to_gyu": {}, "audio_rms": {}, "pairwise_l2_to_student": {}}
    embeddings = {}
    for key, path in paths.items():
        embeddings[key] = wavlm_emb(path)
        metrics["wavlm_to_gyu"][key] = round(float(np.dot(reference_wavlm, embeddings[key])), 5)
        metrics["ecapa_to_gyu"][key] = round(float(np.dot(reference_ecapa, ecapa_emb(path))), 5)
        metrics["audio_rms"][key] = round(float(np.sqrt(np.mean(audio16(path) ** 2))), 6)
    for language in ("ko", "en", "ja"):
        student = embeddings[f"{language}:student"]
        for mode in ("none", "fish_only", "fish_moss", "student"):
            metrics["pairwise_l2_to_student"][f"{language}:{mode}"] = round(float(np.sqrt(np.mean((embeddings[f"{language}:{mode}"] - student) ** 2))), 6)
    report = {
        "paths": paths,
        "metric": "WavLM and ECAPA speaker cosine; same language score/content/F0 per variant",
        "identity_ablation_modes": ["none", "fish_only", "fish_moss", "student"],
        "identity_effect": any(metrics["pairwise_l2_to_student"][f"{language}:none"] > 1e-5 for language in ("ko", "en", "ja")),
        "latent_style_effect": metrics["pairwise_l2_to_student"]["ko:latent_style_zero"] if "ko:latent_style_zero" in metrics["pairwise_l2_to_student"] else True,
        "metrics": metrics,
        "note": "Teacher modes use one held-out-safe paired Fish/MOSS representation as conditioning ablation; student mode uses real-GYU reference projection. Higgs hidden is unavailable.",
    }
    report["three_language_summary"] = {}
    for metric in ("wavlm_to_gyu", "ecapa_to_gyu"):
        none = np.mean([metrics[metric][f"{language}:none"] for language in ("ko", "en", "ja")])
        student = np.mean([metrics[metric][f"{language}:student"] for language in ("ko", "en", "ja")])
        report["three_language_summary"][metric] = {"none_mean": round(float(none), 5), "student_mean": round(float(student), 5), "student_minus_none": round(float(student - none), 5)}
    # Keep style difference explicit; it is measured against the KO student waveform.
    style_key = "ko:latent_style_zero"; student_key = "ko:student"
    report["latent_style_effect"] = float(np.sqrt(np.mean((embeddings[style_key] - embeddings[student_key]) ** 2))) > 1e-5
    Path("artifacts/reports/v06_identity_style_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/soulx_identity_adapter.md").write_text("# v0.6 SoulX identity adapter\n\n" + json.dumps(report, indent=2) + "\n")
    Path("docs/latent_style_adapter.md").write_text("# v0.6 latent acoustic-style adapter\n\n" + json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
