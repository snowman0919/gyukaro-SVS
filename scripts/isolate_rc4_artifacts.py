#!/usr/bin/env python3
"""Render the RC4 A-F artifact matrix and SoulX decoder sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gyu_singer.inference.acoustic_style import adapt_waveform
from gyu_singer.inference.quality_controller import STYLE
from gyu_singer.inference.soulx import _Worker
from gyu_singer.inference.v08 import GyuSingerV08Renderer
from gyu_singer.score import normalize_score


CASES = {
    "ko_neutral": "examples/quality_ko.json",
    "en": "examples/quality_en.json",
    "rapid_ko": "examples/review_rapid_ko.json",
    "large_interval_ko": "examples/review_large_interval_ko.json",
}


def describe(path: Path, seconds: float | None = None) -> dict:
    info = sf.info(path)
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": info.samplerate, "channels": info.channels,
        "duration_seconds": round(info.duration, 4), "render_seconds": None if seconds is None else round(seconds, 3),
    }


def request(worker: _Worker, body: dict, output: Path) -> dict:
    started = time.perf_counter(); worker.request(body | {"output": str(output)})
    return describe(output, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/reports/rc5_isolation")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    args = parser.parse_args()
    root, output = Path.cwd(), Path(args.output); shutil.rmtree(output, ignore_errors=True); output.mkdir(parents=True)
    renderer = GyuSingerV08Renderer(args.reference, root=root)
    prepared = {}
    try:
        for case, score_path in CASES.items():
            directory = output / case; directory.mkdir()
            score = normalize_score(json.loads(Path(score_path).read_text()))
            duration = max(note["start"] + note["duration"] for note in score["notes"])
            raw = directory / "A_omnivoice_source.wav"
            renderer.omnivoice.request({"language": score["language"], "lyrics": "".join(note["lyric"] for note in score["notes"]), "duration": duration, "output": str(raw)})
            expressive = renderer.pitch_controller.predict(score)[0] * score["style"]["prosody_strength"]
            source_duration = sf.info(raw).duration
            nominal = renderer._f0(score, source_duration)
            production = renderer._f0(score, source_duration, expressive.cpu().numpy())
            nominal_path, production_path = directory / "nominal_f0.npy", directory / "production_f0.npy"
            np.save(nominal_path, nominal); np.save(production_path, production)
            identity = renderer._identity_vector()
            style = renderer._style_vector(score["style"], renderer.pitch_controller.device)
            identity_path, style_path = directory / "identity.npy", directory / "style.npy"
            np.save(identity_path, identity.detach().cpu().numpy()); np.save(style_path, style.detach().cpu().numpy())
            controls = np.array([.8, 0, 0, 0, 0], dtype="float32")
            for index, name in enumerate(("dynamics", "breathiness", "tension", "brightness", "vibrato")):
                if score["curves"][name]: controls[index] = float(np.mean([point["value"] for point in score["curves"][name]]))
            identity_ref = renderer.reference_features + .05 * identity.repeat((renderer.reference_features.shape[0] + identity.shape[0] - 1) // identity.shape[0])[:renderer.reference_features.shape[0]]
            preset = torch.tensor(STYLE[renderer._content_style_preset(score["style"])], device=renderer.pitch_controller.device)
            audio, rate = sf.read(raw, dtype="float32", always_2d=True)
            adapted = adapt_waveform(audio.mean(1), rate, renderer.acoustic_adapter, identity_ref, torch.from_numpy(controls).to(renderer.pitch_controller.device), preset, score["style"]["acoustic_style_strength"])
            adapted_path = directory / "production_adapted_source.wav"; sf.write(adapted_path, adapted, rate, subtype="PCM_16")
            prepared[case] = {"raw": raw, "adapted": adapted_path, "nominal": nominal_path, "production": production_path,
                              "identity": identity_path, "style": style_path, "score": score_path}
        renderer.omnivoice.close()

        report = {"status": "complete", "seed": 21, "rc4_decoder": {"n_steps": 16, "cfg": 2.5, "precision": "fp16"}, "cases": {}}
        for case, paths in prepared.items():
            directory = output / case
            common = {"source": str(paths["raw"]), "f0_npy": str(paths["production"]), "n_steps": 16, "cfg": 2.5, "seed": 21}
            matrix = {"A": describe(paths["raw"])}
            matrix["B"] = request(renderer.soulx, common | {"f0_npy": str(paths["nominal"])}, directory / "B_nominal_off_off.wav")
            matrix["C"] = request(renderer.soulx, common, directory / "C_production_f0_off_off.wav")
            matrix["D"] = request(renderer.soulx, common | {"identity_npy": str(paths["identity"])}, directory / "D_identity_on_style_off.wav")
            matrix["E"] = request(renderer.soulx, common | {"style_npy": str(paths["style"])}, directory / "E_identity_off_style_on.wav")
            matrix["F"] = request(renderer.soulx, common | {"source": str(paths["adapted"]), "identity_npy": str(paths["identity"]), "style_npy": str(paths["style"])}, directory / "F_full_rc4.wav")
            sweep = []
            for steps in (16, 32, 64):
                for cfg in (1.5, 2.0, 2.5):
                    target = directory / f"sweep_fp16_s{steps}_c{cfg:g}.wav"
                    if steps == 16 and cfg == 2.5: shutil.copy2(directory / "F_full_rc4.wav", target); row = describe(target, matrix["F"]["render_seconds"])
                    else: row = request(renderer.soulx, common | {"source": str(paths["adapted"]), "identity_npy": str(paths["identity"]), "style_npy": str(paths["style"]), "n_steps": steps, "cfg": cfg}, target)
                    sweep.append({"precision": "fp16", "n_steps": steps, "cfg": cfg} | row)
            report["cases"][case] = {"score": paths["score"], "matrix": matrix, "sweep": sweep,
                                     "f0": {"nominal_voiced_ratio": float(np.mean(np.load(paths["nominal"]) > 0)), "production_voiced_ratio": float(np.mean(np.load(paths["production"]) > 0))}}
        renderer.soulx.close()

        cache = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")); soulx = cache / "soulx-singer"
        command = [str(renderer.soulx_python), "scripts/probe_soulx_score.py", "--worker", "--precision", "fp32", "--reference", str(Path(args.reference).resolve()),
                   "--model", str((soulx / "pretrained_models/SoulX-Singer/model-svc.pt").resolve()), "--config", str((soulx / "soulxsinger/config/soulxsinger.yaml").resolve()),
                   "--rmvpe", str((soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt").resolve()), "--latent-adapter", str((root / "checkpoints/gyu_real_latent_adapters_v0.7.pt").resolve())]
        fp32 = _Worker(command, root, os.environ | {"GYU_SINGER_CACHE": str(cache.resolve())})
        try:
            for case, paths in prepared.items():
                common = {"source": str(paths["adapted"]), "f0_npy": str(paths["production"]), "identity_npy": str(paths["identity"]), "style_npy": str(paths["style"]), "seed": 21}
                for steps in (16, 32, 64):
                    for cfg in (1.5, 2.0, 2.5):
                        target = output / case / f"sweep_fp32_s{steps}_c{cfg:g}.wav"
                        row = request(fp32, common | {"n_steps": steps, "cfg": cfg}, target)
                        report["cases"][case]["sweep"].append({"precision": "fp32", "n_steps": steps, "cfg": cfg} | row)
        finally:
            fp32.close()
        (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(output), "cases": len(report["cases"]), "renders": sum(len(row["matrix"]) + len(row["sweep"]) for row in report["cases"].values())}, indent=2))
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
