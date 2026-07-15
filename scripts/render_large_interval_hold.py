#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, soundfile as sf
from gyu_singer.inference.content_timing import latent_content_hold
from gyu_singer.inference.v09 import GyuSingerV09Renderer

def main():
    rc4, fixed, ctc, out = map(Path, ("artifacts/reports/rc5_isolation", "artifacts/reports/rc5_candidate_core", "artifacts/reports/rc5_content_timing", "artifacts/reports/rc5_latent_hold/large_interval_ko")); out.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((rc4 / "matrix.json").read_text())["cases"]["large_interval_ko"]; source = Path(matrix["matrix"]["A"]["path"]); f0 = np.load(fixed / "large_interval_ko/canonical_f0.npy"); alignment = json.loads((ctc / "large_interval_ko/alignment.json").read_text()); warp = latent_content_hold(alignment, sf.info(source).duration, len(f0)); np.save(out / "content_hold.npy", warp)
    renderer = GyuSingerV09Renderer("data/processed/master/216.wav", root=Path.cwd()); renderer.omnivoice.close(); common = {"source": str(source), "f0_npy": str(fixed / "large_interval_ko/canonical_f0.npy"), "content_warp_npy": str(out / "content_hold.npy"), "n_steps": 64, "cfg": 2.0, "seed": 21}
    try:
        rows = {}
        for label, extra in (("hold_off_off", {}), ("hold_full", {"source": str(rc4 / "large_interval_ko/production_adapted_source.wav"), "identity_npy": str(rc4 / "large_interval_ko/identity.npy"), "style_npy": str(rc4 / "large_interval_ko/style.npy")})):
            target = out / f"{label}.wav"; started = time.perf_counter(); renderer.soulx.request(common | extra | {"output": str(target)}); rows[label] = {"path": str(target), "render_seconds": round(time.perf_counter() - started, 3)}
        (out.parent / "manifest.json").write_text(json.dumps({"status": "rendered_not_reviewed", "case": "large_interval_ko", "renders": rows}, indent=2) + "\n")
    finally: renderer.close()

if __name__ == "__main__": main()
