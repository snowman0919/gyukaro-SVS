#!/usr/bin/env python3
"""Causal v0.6 identity and latent-style ablations on held-out phrase scores."""
import copy
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

from gyu_singer.inference.v05 import GyuSingerV05Renderer
from gyu_singer.inference.v06 import GyuSingerV06Renderer


REPORT = Path("artifacts/reports")
SCORES = [Path(f"examples/{kind}_{language}.json") for language in ("ko", "en", "ja") for kind in ("quality", "heldout")]


def audio16(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
    return resample_poly(audio, 16000, rate).astype("float32") if rate != 16000 else audio


def write(renderer, score, name, paths):
    path = REPORT / f"v06_ablation_{name}.wav"
    sf.write(path, renderer.render(score), renderer.sample_rate, subtype="PCM_16")
    paths[name] = str(path)


def interval(values):
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(606)
    samples = [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
    return {"n": len(values), "mean": round(float(values.mean()), 5), "p2_5": round(float(np.percentile(samples, 2.5)), 5), "p97_5": round(float(np.percentile(samples, 97.5)), 5), "values": [round(float(x), 5) for x in values]}


def centroid(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequency = np.fft.rfftfreq(len(audio), 1 / rate)
    return float((frequency * spectrum).sum() / max(spectrum.sum(), 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=("v05_spectral", "v06_spectral_only", "v06_latent"))
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = {path.stem.removeprefix("v06_ablation_"): str(path) for path in REPORT.glob("v06_ablation_*.wav")}
    if not args.style and not args.metrics_only:
        renderer = GyuSingerV06Renderer("data/processed/master/216.wav")
        try:
            for score_path in SCORES:
                score = json.loads(score_path.read_text()); label = score_path.stem
                for mode in ("none", "fish_only", "fish_moss", "student"):
                    renderer.identity_mode, renderer.identity_enabled, renderer.style_enabled = mode, mode != "none", True
                    write(renderer, score, f"identity_{mode}_{label}", paths)
        finally:
            renderer.close()
        return
    factories = {"v05_spectral": GyuSingerV05Renderer, "v06_spectral_only": lambda ref: GyuSingerV06Renderer(ref, latent_adapter_enabled=False), "v06_latent": GyuSingerV06Renderer}
    if args.style:
        name, factory = args.style, factories[args.style]
        renderer = factory("data/processed/master/216.wav")
        try:
            for score_path in (Path("examples/quality_ko.json"), Path("examples/heldout_ko.json")):
                base = json.loads(score_path.read_text())
                for preset in ("neutral", "dark"):
                    score = copy.deepcopy(base); score.setdefault("style", {})["preset"] = preset
                    if hasattr(renderer, "identity_mode"):
                        renderer.identity_mode, renderer.identity_enabled, renderer.style_enabled = "student", True, name == "v06_latent"
                    write(renderer, score, f"style_{name}_{preset}_{score_path.stem}", paths)
        finally:
            renderer.close()

    if not args.metrics_only:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    ecapa = EncoderClassifier.from_hparams(source="data/cache/spkrec-ecapa-voxceleb", savedir="data/cache/spkrec-ecapa-voxceleb", run_opts={"device": device})
    def wavlm_emb(path):
        value = extractor(audio16(path), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): value = wavlm(**{key: item.to(device) for key, item in value.items()}).embeddings
        return torch.nn.functional.normalize(value, dim=-1).squeeze().cpu().numpy()
    def ecapa_emb(path):
        with torch.inference_mode(): value = ecapa.encode_batch(torch.from_numpy(audio16(path)).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        return value / max(np.linalg.norm(value), 1e-8)
    reference = (wavlm_emb("data/processed/master/216.wav"), ecapa_emb("data/processed/master/216.wav"))
    embs = {key: (wavlm_emb(path), ecapa_emb(path)) for key, path in paths.items()}
    identity = {}
    for score_path in SCORES:
        label = score_path.stem; language = label.rsplit("_", 1)[1]
        variants = {}
        for mode in ("none", "fish_only", "fish_moss", "student"):
            key = f"identity_{mode}_{label}"; w, e = embs[key]
            variants[mode] = {"wavlm_to_gyu": round(float(np.dot(reference[0], w)), 5), "ecapa_to_gyu": round(float(np.dot(reference[1], e)), 5), "wavlm_l2_to_student": round(float(np.sqrt(np.mean((w - embs[f'identity_student_{label}'][0]) ** 2))), 6)}
        identity[label] = {"language": language, "variants": variants}
    deltas = {metric: [identity[label]["variants"]["student"][metric] - identity[label]["variants"]["none"][metric] for label in identity] for metric in ("wavlm_to_gyu", "ecapa_to_gyu")}
    style = {}
    for score_path in (Path("examples/quality_ko.json"), Path("examples/heldout_ko.json")):
        label = score_path.stem; style[label] = {}
        for name in ("v05_spectral", "v06_spectral_only", "v06_latent"):
            neutral = embs[f"style_{name}_neutral_{label}"][0]; dark = embs[f"style_{name}_dark_{label}"][0]
            style[label][name] = {"dark_minus_neutral_wavlm_l2": round(float(np.sqrt(np.mean((dark - neutral) ** 2))), 6), "dark_minus_neutral_centroid_hz": round(centroid(paths[f"style_{name}_dark_{label}"]) - centroid(paths[f"style_{name}_neutral_{label}"]), 3)}
        style[label]["latent_vs_spectral_only_dark_wavlm_l2"] = round(float(np.sqrt(np.mean((embs[f"style_v06_latent_dark_{label}"][0] - embs[f"style_v06_spectral_only_dark_{label}"][0]) ** 2))), 6)
    report = {"paths": paths, "identity_ablation": {"scores": identity, "student_minus_no_identity": {metric: interval(values) for metric, values in deltas.items()}}, "style_ablation": style, "protocol": "Fixed score, lyrics, deterministic OmniVoice/SoulX seeds, and reference; only the named conditioning is varied. Identity uses two phrases per KO/EN/JA. Style compares v0.5 spectral-only, v0.6 without latent injection, and v0.6 latent injection on two KO phrases.", "caveat": "This measures output effects, not a claim that the weak teacher-style supervision establishes every style preset semantically."}
    Path("artifacts/reports/v06_identity_style_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/soulx_identity_adapter.md").write_text("# v0.6 SoulX identity adapter\n\n" + json.dumps(report["identity_ablation"], indent=2) + "\n\n" + report["protocol"] + "\n")
    Path("docs/latent_style_adapter.md").write_text("# v0.6 latent acoustic-style adapter\n\n" + json.dumps({"style_ablation": style, "caveat": report["caveat"]}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
