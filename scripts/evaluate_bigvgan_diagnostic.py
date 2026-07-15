#!/usr/bin/env python3
"""Official BigVGAN resynthesis diagnostic for identical RC4/RC5 stress audio."""
from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch


CACHE = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache"))
BIGVGAN_REVISION = "7d2b454564a6c7d014227f635b7423881f14bdac"
BIGVGAN_MODEL_REVISION = "c329ede9e9bbc100ddf5c91e2330a61921262370"
SCORES = {"ko_neutral": "examples/quality_ko.json", "en": "examples/quality_en.json", "rapid_ko": "examples/review_rapid_ko.json", "large_interval_ko": "examples/review_large_interval_ko.json"}


def load_bigvgan(device: str):
    sys.path.insert(0, str(CACHE / "bigvgan"))
    import bigvgan
    model = bigvgan.BigVGAN._from_pretrained(model_id=str(CACHE / "bigvgan-checkpoint"), revision=BIGVGAN_MODEL_REVISION, cache_dir=None, force_download=False, proxies=None, resume_download=False, local_files_only=True, token=None, use_cuda_kernel=False)
    model.remove_weight_norm(); return model.eval().to(device)


def resynthesize(model, source: Path, target: Path, device: str) -> None:
    from meldataset import get_mel_spectrogram
    audio, rate = sf.read(source, dtype="float32", always_2d=True); mono = audio.mean(1)
    wav = resample_poly(mono, model.h.sampling_rate, rate).astype("float32") if rate != model.h.sampling_rate else mono
    with torch.inference_mode():
        tensor = torch.from_numpy(wav).to(device)[None]
        output = model(get_mel_spectrogram(tensor, model.h)).squeeze().float().cpu().numpy()
    output = resample_poly(output, 48_000, model.h.sampling_rate).astype("float32")
    peak = float(np.max(np.abs(output))); output *= min(1.0, .97 / max(peak, 1e-8))
    sf.write(target, output, 48_000, subtype="PCM_24")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="artifacts/reports/bigvgan_diagnostic"); args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    rc4 = json.loads(Path("artifacts/reports/rc5_isolation/matrix.json").read_text())
    rc5 = json.loads(Path("artifacts/reports/rc5_stress_candidate4/manifest.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = load_bigvgan(device)
    sources = []
    for case in SCORES:
        for version, path in (("rc4", rc4["cases"][case]["matrix"]["F"]["path"]), ("rc5", rc5["files"][case]["path"])):
            source, output = Path(path), root / f"{case}_{version}_bigvgan.wav"
            resynthesize(model, source, output, device); sources.append((case, version, source, output))
    del model; torch.cuda.empty_cache()

    sys.path.insert(0, str(CACHE / "soulx-singer"))
    from preprocess.tools.f0_extraction import F0Extractor
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device=device, target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for case, version, source, output in sources:
        target = np.load(Path("artifacts/reports/rc5_stress_candidate4") / f"{case}_target_f0.npy")
        rows.append({"case": case, "version": version, "variant": "input", "path": str(source)} | acoustics(source) | pitch(source, target, extractor))
        rows.append({"case": case, "version": version, "variant": "bigvgan", "path": str(output)} | acoustics(output) | pitch(output, target, extractor))
    del extractor; torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).to(device).eval()
    for row in rows:
        score = json.loads(Path(SCORES[row["case"]]).read_text()); expected = normalized(" ".join(note["lyric"] for note in score["notes"]))
        inputs = processor(audio16(Path(row["path"])), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode(): ids = asr.generate(inputs.input_features.to(device).half(), language=score["language"], task="transcribe", max_new_tokens=64)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]; row["asr_transcript"] = transcript
        row["asr_lyric_similarity"] = round(SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4)
    metrics = ("pitch_mae_cents", "voicing_accuracy", "hf_energy_ratio_p95", "hf_spike_p99_over_median", "spectral_flatness_mean", "spectral_flux_p95", "sample_jump_p999", "asr_lyric_similarity")
    def aggregate(version: str, variant: str) -> dict:
        selected = [row for row in rows if row["version"] == version and row["variant"] == variant]
        return {name: round(float(np.mean([row[name] for row in selected if row[name] is not None])), 6) for name in metrics}
    aggregates = {version: {variant: aggregate(version, variant) for variant in ("input", "bigvgan")} for version in ("rc4", "rc5")}
    deltas = {version: {name: round(aggregates[version]["bigvgan"][name] - aggregates[version]["input"][name], 6) for name in metrics} for version in aggregates}
    report = {"status": "diagnostic_not_selected", "model": "official NVIDIA BigVGAN v2 24kHz 100-band 256x", "code_revision": BIGVGAN_REVISION, "model_revision": BIGVGAN_MODEL_REVISION, "license": "MIT", "use_cuda_kernel": False, "waveform_pitch_control": False, "aggregates": aggregates, "bigvgan_minus_input": deltas, "rows": rows, "decision_rule": "promotion requires lower artifacts without material pitch, ASR, consonant, or listening regression"}
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"aggregates": aggregates, "bigvgan_minus_input": deltas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
