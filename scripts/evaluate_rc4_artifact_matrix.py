#!/usr/bin/env python3
"""Measure RC4 A-F and SoulX sweep outputs without claiming perceptual MOS."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

CACHE = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache"))
sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor


def normalized(text: str) -> str:
    return re.sub(r"[^a-zA-Z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def audio16(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); mono = audio.mean(1)
    return librosa.resample(mono, orig_sr=rate, target_sr=16000) if rate != 16000 else mono


def acoustics(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); y = audio.mean(1)
    if rate != 48000:
        y = librosa.resample(y, orig_sr=rate, target_sr=48000); rate = 48000
    frequencies, _, spectrum = stft(y, fs=rate, nperseg=1024, noverlap=768, boundary=None)
    power = np.abs(spectrum).T ** 2 + 1e-12
    normalized_spectrum = power / power.sum(1, keepdims=True)
    hf = normalized_spectrum[:, frequencies >= min(8000, rate / 2 * .8)].sum(1)
    flux = np.sqrt(np.sum(np.diff(normalized_spectrum, axis=0) ** 2, axis=1))
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    jumps = np.abs(np.diff(y))
    return {
        "peak": round(float(np.max(np.abs(y))), 6), "rms": round(float(np.sqrt(np.mean(y * y))), 6),
        "clip_fraction": round(float(np.mean(np.abs(y) >= .999)), 8),
        "hf_energy_ratio_mean": round(float(np.mean(hf)), 6), "hf_energy_ratio_p95": round(float(np.percentile(hf, 95)), 6),
        "hf_spike_p99_over_median": round(float(np.percentile(hf, 99) / max(np.median(hf), 1e-8)), 4),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 6),
        "spectral_flux_p95": round(float(np.percentile(flux, 95)), 6),
        "spectral_flux_max": round(float(np.max(flux)), 6),
        "sample_jump_p999": round(float(np.percentile(jumps, 99.9)), 6),
    }


def pitch(path: Path, target: np.ndarray | None, extractor) -> dict:
    observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
    result = {"observed_voiced_ratio": round(float(np.mean(observed > 1)), 4)}
    if target is None: return result
    target = np.interp(np.arange(len(observed)), np.linspace(0, len(observed) - 1, len(target)), target)
    both = (observed > 1) & (target > 1)
    result["target_voiced_ratio"] = round(float(np.mean(target > 1)), 4)
    result["voicing_accuracy"] = round(float(np.mean((observed > 1) == (target > 1))), 4)
    result["pitch_mae_cents"] = None if not both.any() else round(float(np.median(np.abs(1200 * np.log2(observed[both] / target[both])))), 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="artifacts/reports/rc5_isolation")
    args = parser.parse_args(); root = Path(args.root); matrix = json.loads((root / "matrix.json").read_text())
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for case, case_data in matrix["cases"].items():
        nominal = np.load(root / case / "nominal_f0.npy"); production = np.load(root / case / "production_f0.npy")
        for label, item in case_data["matrix"].items():
            path = Path(item["path"]); target = None if label == "A" else nominal if label == "B" else production
            rows.append({"case": case, "group": "matrix", "label": label, "path": str(path)} | acoustics(path) | pitch(path, target, extractor))
        for item in case_data["sweep"]:
            path = Path(item["path"])
            rows.append({"case": case, "group": "sweep", "label": f"{item['precision']}_s{item['n_steps']}_c{item['cfg']:g}",
                         "precision": item["precision"], "n_steps": item["n_steps"], "cfg": item["cfg"], "render_seconds": item["render_seconds"], "path": str(path)} |
                        acoustics(path) | pitch(path, production, extractor))
    del extractor; torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    scores = {case: json.loads(Path(data["score"]).read_text()) for case, data in matrix["cases"].items()}
    for row in rows:
        if row["group"] != "matrix": continue
        expected = " ".join(note["lyric"] for note in scores[row["case"]]["notes"])
        inputs = processor(audio16(Path(row["path"])), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(inputs.input_features.cuda().half(), language=scores[row["case"]]["language"], task="transcribe", max_new_tokens=64)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_transcript"] = transcript
        row["asr_lyric_similarity"] = round(SequenceMatcher(None, normalized(expected), normalized(transcript)).ratio(), 4)

    metrics = ("pitch_mae_cents", "observed_voiced_ratio", "hf_energy_ratio_p95", "hf_spike_p99_over_median", "spectral_flatness_mean", "spectral_flux_p95", "sample_jump_p999")
    def aggregate(selected: list[dict]) -> dict:
        return {name: round(float(np.mean([row[name] for row in selected if row.get(name) is not None])), 6) for name in metrics}
    stage = {label: aggregate([row for row in rows if row["group"] == "matrix" and row["label"] == label]) for label in "ABCDEF"}
    sweep = []
    labels = sorted({row["label"] for row in rows if row["group"] == "sweep"})
    for label in labels:
        selected = [row for row in rows if row["group"] == "sweep" and row["label"] == label]
        aggregate_row = aggregate(selected)
        sweep.append({"label": label, "precision": selected[0]["precision"], "n_steps": selected[0]["n_steps"], "cfg": selected[0]["cfg"],
                      "mean_render_seconds": round(float(np.mean([row["render_seconds"] for row in selected])), 3)} | aggregate_row)
    for row in sweep:
        row["artifact_proxy"] = round(row["hf_energy_ratio_p95"] + row["spectral_flatness_mean"] + row["spectral_flux_p95"], 6)
    sweep.sort(key=lambda row: (row["artifact_proxy"], row["pitch_mae_cents"]))

    listening = root / "listening_matrix"; shutil.rmtree(listening, ignore_errors=True); listening.mkdir()
    for row in rows:
        if row["group"] == "matrix": shutil.copy2(row["path"], listening / f"{row['case']}_{row['label']}.wav")
    for candidate in sweep[:3]:
        for row in rows:
            if row["group"] == "sweep" and row["label"] == candidate["label"]:
                shutil.copy2(row["path"], listening / f"{row['case']}_{row['label']}.wav")
    report = {
        "status": "measured_not_human_reviewed", "metric_warning": "artifact proxies and ASR are diagnostic only; they cannot pass release listening",
        "known_structural_defect": "RC4 target F0 is nonzero for every frame, including unvoiced consonants and silence",
        "stage_aggregate": stage, "decoder_sweep_ranked_by_proxy": sweep, "rows": rows,
        "listening_directory": str(listening),
    }
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage_aggregate": stage, "top_decoder_proxy_candidates": sweep[:5], "listening_directory": str(listening)}, indent=2))


if __name__ == "__main__": main()
