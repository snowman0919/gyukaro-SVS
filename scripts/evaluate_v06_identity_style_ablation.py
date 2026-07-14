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
from gyu_singer.inference.v07 import GyuSingerV07Renderer


REPORT = Path("artifacts/reports")
SCORES = [Path(f"examples/{kind}_{language}.json") for language in ("ko", "en", "ja") for kind in ("quality", "heldout")]


def audio16(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
    return resample_poly(audio, 16000, rate).astype("float32") if rate != 16000 else audio


def write(renderer, score, name, paths, version="v06"):
    path = REPORT / f"{version}_ablation_{name}.wav"
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
    parser.add_argument("--version", choices=("v06", "v07"), default="v06")
    parser.add_argument("--style", choices=("v05_spectral", "v06_spectral_only", "v06_latent", "v07_latent"))
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for version in ("v06", "v07"):
        paths.update({path.stem.removeprefix(f"{version}_ablation_"): str(path) for path in REPORT.glob(f"{version}_ablation_*.wav")})
    if not args.style and not args.metrics_only:
        renderer = (GyuSingerV07Renderer if args.version == "v07" else GyuSingerV06Renderer)("data/processed/master/216.wav")
        try:
            for score_path in SCORES:
                score = json.loads(score_path.read_text()); label = score_path.stem
                for mode in ("none", "fish_only", "fish_moss", "student"):
                    renderer.identity_mode, renderer.identity_enabled, renderer.style_enabled = mode, mode != "none", True
                    write(renderer, score, f"identity_{mode}_{label}", paths, args.version)
        finally:
            renderer.close()
        return
    factories = {"v05_spectral": GyuSingerV05Renderer, "v06_spectral_only": lambda ref: GyuSingerV06Renderer(ref, latent_adapter_enabled=False), "v06_latent": GyuSingerV06Renderer, "v07_latent": GyuSingerV07Renderer}
    if args.style:
        name, factory = args.style, factories[args.style]
        renderer = factory("data/processed/master/216.wav")
        try:
            for score_path in (Path("examples/quality_ko.json"), Path("examples/heldout_ko.json")):
                base = json.loads(score_path.read_text())
                for preset in ("neutral", "dark"):
                    score = copy.deepcopy(base); score.setdefault("style", {})["preset"] = preset
                    if hasattr(renderer, "identity_mode"):
                        renderer.identity_mode, renderer.identity_enabled, renderer.style_enabled = "student", True, name in {"v06_latent", "v07_latent"}
                    write(renderer, score, f"style_{name}_{preset}_{score_path.stem}", paths, args.version)
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
    cross_language = {}
    for kind in ("quality", "heldout"):
        cross_language[kind] = {}
        for mode in ("none", "fish_only", "fish_moss", "student"):
            values = [embs[f"identity_{mode}_{kind}_{language}"][0] for language in ("ko", "en", "ja")]
            pairs = [float(np.dot(values[left], values[right])) for left, right in ((0, 1), (0, 2), (1, 2))]
            cross_language[kind][mode] = {"wavlm_pairwise_mean": round(float(np.mean(pairs)), 5), "pairs": [round(value, 5) for value in pairs]}
    style = {}
    for score_path in (Path("examples/quality_ko.json"), Path("examples/heldout_ko.json")):
        label = score_path.stem; style[label] = {}
        style_names = ("v05_spectral", "v06_spectral_only", "v06_latent", "v07_latent") if args.version == "v07" else ("v05_spectral", "v06_spectral_only", "v06_latent")
        for name in style_names:
            neutral = embs[f"style_{name}_neutral_{label}"][0]; dark = embs[f"style_{name}_dark_{label}"][0]
            style[label][name] = {"dark_minus_neutral_wavlm_l2": round(float(np.sqrt(np.mean((dark - neutral) ** 2))), 6), "dark_minus_neutral_centroid_hz": round(centroid(paths[f"style_{name}_dark_{label}"]) - centroid(paths[f"style_{name}_neutral_{label}"]), 3)}
        comparison = "v07_latent" if args.version == "v07" else "v06_latent"
        style[label]["latent_vs_spectral_only_dark_wavlm_l2"] = round(float(np.sqrt(np.mean((embs[f"style_{comparison}_dark_{label}"][0] - embs[f"style_v06_spectral_only_dark_{label}"][0]) ** 2))), 6)
    report = {"paths": paths, "identity_ablation": {"scores": identity, "student_minus_no_identity": {metric: interval(values) for metric, values in deltas.items()}, "cross_language_identity_consistency": cross_language}, "style_ablation": style, "protocol": f"Fixed score, lyrics, deterministic OmniVoice/SoulX seeds, and reference; only named conditioning varies. {args.version} identity uses two phrases per KO/EN/JA. Style compares available spectral and latent paths on two KO phrases.", "caveat": "Output differences are evidence only; intended semantic direction requires separate acoustic-proxy checks."}
    Path(f"artifacts/reports/{args.version}_identity_style_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    identity_doc = "docs/identity_adapter_v0.7.md" if args.version == "v07" else "docs/soulx_identity_adapter.md"
    style_doc = "docs/style_adapter_v0.7.md" if args.version == "v07" else "docs/latent_style_adapter.md"
    Path(identity_doc).write_text(f"# {args.version} SoulX identity adapter\n\n" + json.dumps(report["identity_ablation"], indent=2) + "\n\n" + report["protocol"] + "\n")
    if args.version != "v07":
        Path(style_doc).write_text(f"# {args.version} latent acoustic-style adapter\n\n" + json.dumps({"style_ablation": style, "caveat": report["caveat"]}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
