#!/usr/bin/env python3
"""Isolate sustained-note noise across SoulX decode settings before RC8 changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from analyze_rc8_defects import metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime  # noqa: E402
from gyu_singer.inference.acoustic_style import adapt_waveform  # noqa: E402
from gyu_singer.inference.quality_controller import STYLE  # noqa: E402
from gyu_singer.inference.spectral_refiner import SpectralRefinerRuntime  # noqa: E402
from gyu_singer.inference.v09 import GyuSingerV09Renderer  # noqa: E402
from gyu_singer.score import normalize_score  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


CASES = {
    "sustained_ko": "examples/review_sustain_ko.json",
    "en": "examples/quality_en.json",
    "ja": "examples/quality_ja.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--case", choices=tuple(CASES), default="sustained_ko")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"artifacts/reports/rc8_{args.case}_decode_sweep"
    listening = output / "listening"
    raw_output = output / "raw_soulx"
    listening.mkdir(parents=True, exist_ok=True)
    raw_output.mkdir(parents=True, exist_ok=True)
    score_path = ROOT / CASES[args.case]
    score = normalize_score(json.loads(score_path.read_text()))
    reference = args.runtime_root / "data/processed/master/216.wav"
    renderer = GyuSingerV09Renderer(reference, root=args.runtime_root)
    raw_content = output / "omnivoice_source.wav"
    content = output / "fixed_content.wav"
    contour = output / "fixed_f0.npy"
    identity_path = output / "fixed_identity.npy"
    style_path = output / "fixed_style.npy"
    try:
        duration = max(note["start"] + note["duration"] for note in score["notes"])
        if not raw_content.is_file():
            renderer.omnivoice.request({
                "language": score["language"], "lyrics": "".join(note["lyric"] for note in score["notes"]),
                "duration": duration, "output": str(raw_content),
            })
        renderer.omnivoice.close()
        expressive = renderer.pitch_controller.predict(score, canonical_timing=True)[0] * score["style"]["prosody_strength"]
        content_audio, content_rate = sf.read(raw_content, dtype="float32", always_2d=True)
        content_audio = content_audio.mean(1)
        identity = renderer._identity_vector()
        style_vector = renderer._style_vector(score["style"], renderer.pitch_controller.device)
        identity_ref = renderer.reference_features + .05 * identity.repeat(
            (renderer.reference_features.shape[0] + identity.shape[0] - 1) // identity.shape[0]
        )[:renderer.reference_features.shape[0]]
        controls = np.array([.8, 0, 0, 0, 0], dtype="float32")
        for index, name in enumerate(("dynamics", "breathiness", "tension", "brightness", "vibrato")):
            if score["curves"][name]:
                controls[index] = float(np.mean([point["value"] for point in score["curves"][name]]))
        preset = torch.tensor(STYLE[renderer._content_style_preset(score["style"])], device=renderer.pitch_controller.device)
        content_audio = adapt_waveform(
            content_audio, content_rate, renderer.acoustic_adapter, identity_ref,
            torch.from_numpy(controls).to(renderer.pitch_controller.device), preset,
            score["style"]["acoustic_style_strength"],
        )
        sf.write(content, content_audio, content_rate, subtype="PCM_16")
        source_duration = len(content_audio) / content_rate
        target_f0, timeline = renderer._canonical_f0(score, source_duration, expressive.cpu().numpy())
        content_options = renderer._content_options(score, content, target_f0, output)
        np.save(contour, target_f0)
        np.save(identity_path, identity.detach().cpu().numpy())
        np.save(style_path, style_vector.detach().cpu().numpy())
        (output / "timeline.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in timeline)
        )
        rows = []
        for steps in (32, 50, 64):
            for cfg in (1.5, 2.0):
                label = f"s{steps}_c{cfg:g}"
                raw_path = raw_output / f"{label}.wav"
                started = time.perf_counter()
                if not raw_path.is_file():
                    renderer.soulx.request({
                        "source": str(content), "f0_npy": str(contour),
                        "identity_npy": str(identity_path), "style_npy": str(style_path),
                        "n_steps": steps, "cfg": cfg, "seed": 21, "output": str(raw_path),
                        **content_options,
                    })
                rows.append({
                    "variant": label, "n_steps": steps, "cfg": cfg,
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "render_seconds": round(time.perf_counter() - started, 3),
                })
    finally:
        renderer.close()

    waveform_refiner = AcousticRefinerRuntime(
        args.runtime_root / "checkpoints/acoustic_refiner_universal.pt", device="cuda",
    )
    spectral_refiner = SpectralRefinerRuntime(
        args.runtime_root / "checkpoints/acoustic_refiner_spectral_singing.pt", device="cuda",
    )
    for row in rows:
        raw, rate = sf.read(ROOT / row["raw_path"], dtype="float32", always_2d=True)
        audio = raw.mean(1)
        if rate != 48_000:
            audio = resample_poly(audio, 48_000, rate).astype("float32")
            rate = 48_000
        waveform = waveform_refiner.process(audio)
        audio = audio + .25 * (waveform - audio)
        spectral = spectral_refiner.process(audio)
        audio = audio + .5 * (spectral - audio)
        audio *= min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))
        path = listening / f"{row['variant']}.wav"
        sf.write(path, audio, rate, subtype="PCM_24")
        row["path"] = str(path.relative_to(ROOT))
        row["sha256"] = digest(path)

    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
    )
    for row in rows:
        path = ROOT / row["path"]
        raw_path = ROOT / row["raw_path"]
        row.update(acoustics(path) | pitch(path, target_f0, extractor) | metrics(path, score_path))
        row["raw_soulx_metrics"] = acoustics(raw_path) | metrics(raw_path, score_path)
    del extractor, waveform_refiner, spectral_refiner
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", dtype=torch.float16,
    ).cuda().eval()
    expected = normalized("".join(note["lyric"] for note in score["notes"]))
    for row in rows:
        inputs = processor(audio16(ROOT / row["path"]), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(
                inputs.input_features.cuda().half(), language=score["language"], task="transcribe",
                max_new_tokens=32,
            )
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_lyric_similarity"] = round(
            SequenceMatcher(None, expected, normalized(row["asr_transcript"])).ratio(), 4,
        )

    baseline = next(row for row in rows if row["variant"] == "s64_c2")
    eligible = []
    for row in rows:
        if row is baseline:
            continue
        if (
            row["asr_lyric_similarity"] >= baseline["asr_lyric_similarity"]
            and row["pitch_mae_cents"] <= baseline["pitch_mae_cents"] + 2
            and row["voicing_accuracy"] >= baseline["voicing_accuracy"] - .01
            and row["rms"] >= .95 * baseline["rms"]
            and row["harmonic_to_noise_proxy_db"] >= baseline["harmonic_to_noise_proxy_db"]
            and all(
                row["multi_resolution"][resolution]["spectral_instability_p95"]
                <= baseline["multi_resolution"][resolution]["spectral_instability_p95"]
                for resolution in ("short", "medium", "long")
            )
        ):
            eligible.append(row)
    selected = None if not eligible else min(
        eligible,
        key=lambda row: sum(
            row["multi_resolution"][resolution]["noise_floor_modulation_db_std"]
            for resolution in ("short", "medium", "long")
        ),
    )
    report = {
        "status": "decoder_hypothesis_supported_human_pending" if selected else "decoder_hypothesis_rejected",
        "baseline": "s64_c2 matches the frozen RC7 decoder policy; all inputs and post-refiners are fixed",
        "selection": None if selected is None else selected["variant"],
        "rows": rows,
        "human_listening": "not_performed",
    }
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "selection": report["selection"],
        "rows": [
            {
                "variant": row["variant"], "seconds": row["render_seconds"],
                "asr": row["asr_transcript"], "pitch": row["pitch_mae_cents"],
                "voicing": row["voicing_accuracy"], "rms": row["rms"],
                "hnr": row["harmonic_to_noise_proxy_db"],
                "instability": {
                    resolution: row["multi_resolution"][resolution]["spectral_instability_p95"]
                    for resolution in ("short", "medium", "long")
                },
            }
            for row in rows
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
