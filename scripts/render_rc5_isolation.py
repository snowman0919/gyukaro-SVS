#!/usr/bin/env python3
"""Render the post-RC4 canonical-timeline candidate against the frozen matrix."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from gyu_singer.inference.v09 import GyuSingerV09Renderer
from gyu_singer.score import normalize_score


def render(worker, body: dict, path: Path) -> dict:
    started = time.perf_counter(); worker.request(body | {"output": str(path)})
    return {"path": str(path), "render_seconds": round(time.perf_counter() - started, 3), "duration_seconds": round(sf.info(path).duration, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rc4", default="artifacts/reports/rc5_isolation"); parser.add_argument("--output", default="artifacts/reports/rc5_candidate_core")
    args = parser.parse_args(); rc4, output = Path(args.rc4), Path(args.output); shutil.rmtree(output, ignore_errors=True); output.mkdir(parents=True)
    frozen = json.loads((rc4 / "matrix.json").read_text()); renderer = GyuSingerV09Renderer("data/processed/master/216.wav", root=Path.cwd())
    report = {"status": "rendered_not_human_reviewed", "baseline": "v1.0.0-rc.4", "candidate": "post-RC4 v0.9 engineering path", "decoder": {"n_steps": 64, "cfg": 2.0, "precision": "fp32"}, "cases": {}}
    try:
        renderer.omnivoice.close()
        for case, old in frozen["cases"].items():
            directory = output / case; directory.mkdir()
            score = normalize_score(json.loads(Path(old["score"]).read_text()))
            raw = Path(old["matrix"]["A"]["path"]); duration = sf.info(raw).duration
            expressive = renderer.pitch_controller.predict(score, canonical_timing=True)[0] * score["style"]["prosody_strength"]
            f0, timeline = renderer._canonical_f0(score, duration, expressive.cpu().numpy())
            contour = directory / "canonical_f0.npy"; np.save(contour, f0)
            (directory / "timeline.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in timeline))
            common = {"source": str(raw), "f0_npy": str(contour), "n_steps": 64, "cfg": 2.0, "seed": 21}
            identity, style = rc4 / case / "identity.npy", rc4 / case / "style.npy"
            rows = {
                "rc4": {"path": old["matrix"]["F"]["path"]},
                "fixed_off_off": render(renderer.soulx, common, directory / "fixed_off_off.wav"),
                "fixed_identity_only": render(renderer.soulx, common | {"identity_npy": str(identity)}, directory / "fixed_identity_only.wav"),
                "fixed_style_only": render(renderer.soulx, common | {"style_npy": str(style)}, directory / "fixed_style_only.wav"),
                "fixed_full": render(renderer.soulx, common | {"source": str(rc4 / case / "production_adapted_source.wav"), "identity_npy": str(identity), "style_npy": str(style)}, directory / "fixed_full.wav"),
            }
            report["cases"][case] = {"score": old["score"], "target_voiced_ratio_rc4": old["f0"]["production_voiced_ratio"], "target_voiced_ratio_fixed": round(float(np.mean(f0 > 0)), 4), "voicing_counts": {name: sum(row["voicing"] == name for row in timeline) for name in ("vowel", "voiced_consonant", "unvoiced_consonant", "silence")}, "renders": rows}
        (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(output), "cases": len(report["cases"]), "voiced_ratios": {case: row["target_voiced_ratio_fixed"] for case, row in report["cases"].items()}}, indent=2))
    finally:
        renderer.close()


if __name__ == "__main__": main()
