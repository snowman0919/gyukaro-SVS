#!/usr/bin/env python3
"""Render CTC-warped SoulX content hidden states on the canonical score grid."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from gyu_singer.inference.content_timing import latent_content_warp
from gyu_singer.inference.v09 import GyuSingerV09Renderer


def call(worker, body: dict, output: Path) -> dict:
    started = time.perf_counter(); worker.request(body | {"output": str(output)}); return {"path": str(output), "render_seconds": round(time.perf_counter() - started, 3)}


def main() -> None:
    rc4, fixed, ctc, output = map(Path, ("artifacts/reports/rc5_isolation", "artifacts/reports/rc5_candidate_core", "artifacts/reports/rc5_content_timing", "artifacts/reports/rc5_latent_timing")); shutil.rmtree(output, ignore_errors=True); output.mkdir(parents=True)
    matrix = json.loads((rc4 / "matrix.json").read_text()); renderer = GyuSingerV09Renderer("data/processed/master/216.wav", root=Path.cwd()); report = {"status": "rendered_not_human_reviewed", "method": "MMS CTC mapping applied to SoulX Whisper hidden only", "waveform_time_stretch": False, "waveform_pitch_shift": False, "cases": {}}
    try:
        renderer.omnivoice.close()
        for case, data in matrix["cases"].items():
            directory = output / case; directory.mkdir(); source = Path(data["matrix"]["A"]["path"]); duration = sf.info(source).duration; f0 = np.load(fixed / case / "canonical_f0.npy")
            alignment = json.loads((ctc / case / "alignment.json").read_text()); warp = latent_content_warp(alignment, duration, max(row["target_end"] for row in alignment["phones"]), len(f0)); warp_path = directory / "content_warp.npy"; np.save(warp_path, warp)
            common = {"source": str(source), "f0_npy": str(fixed / case / "canonical_f0.npy"), "content_warp_npy": str(warp_path), "n_steps": 64, "cfg": 2.0, "seed": 21}
            identity, style = rc4 / case / "identity.npy", rc4 / case / "style.npy"
            rows = {"latent_timing_off_off": call(renderer.soulx, common, directory / "latent_timing_off_off.wav"),
                    "latent_timing_full": call(renderer.soulx, common | {"source": str(rc4 / case / "production_adapted_source.wav"), "identity_npy": str(identity), "style_npy": str(style)}, directory / "latent_timing_full.wav")}
            report["cases"][case] = {"score": data["score"], "warp": str(warp_path), "source_duration": duration, "renders": rows}
        (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({"output": str(output), "cases": len(report["cases"])}, indent=2))
    finally: renderer.close()


if __name__ == "__main__": main()
