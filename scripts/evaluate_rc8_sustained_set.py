#!/usr/bin/env python3
"""Evaluate sustained-noise strengths on six fixed RC7-policy phrases."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from analyze_rc8_defects import metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, pitch  # noqa: E402
from gyu_singer.data import acoustic_reference_features  # noqa: E402
from gyu_singer.inference.quality_controller import QualityPitchController  # noqa: E402
from gyu_singer.inference.soulx import SoulXPhraseRenderer  # noqa: E402
from gyu_singer.score import normalize_score  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_sustained_set"
    manifest = json.loads((root / "manifest.json").read_text())
    gated = json.loads((ROOT / "artifacts/reports/rc8_stationary_gate/manifest.json").read_text())["sustained"]
    items = manifest["rows"] + [{"case": case, "score": row["score"], "spectral_strength": "stationary_gate", "path": row["path"]} for case, row in gated.items()]
    controller = QualityPitchController(ROOT / "checkpoints/gyu_prosody_v0.5.pt", acoustic_reference_features(ROOT / "data/processed/master/216.wav"))
    targets = {}
    for row in items:
        if row["case"] in targets:
            continue
        score = normalize_score(json.loads((ROOT / row["score"]).read_text()))
        duration = sf.info(ROOT / row["path"]).duration
        expressive = controller.predict(score, canonical_timing=True)[0] * score["style"]["prosody_strength"]
        targets[row["case"]], _ = SoulXPhraseRenderer._canonical_f0(score, duration, expressive.cpu().numpy())
    del controller
    torch.cuda.empty_cache()
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for item in items:
        path = ROOT / item["path"]
        rows.append(item | acoustics(path) | pitch(path, targets[item["case"]], extractor) | metrics(path, ROOT / item["score"]))
    del extractor
    torch.cuda.empty_cache()
    keys = ("pitch_mae_cents", "voicing_accuracy", "hf_spike_p99_over_median", "spectral_flux_p95", "sample_jump_p999", "harmonic_to_noise_proxy_db")
    aggregate = defaultdict(dict)
    for strength in manifest["spectral_strengths"] + ["stationary_gate"]:
        selected = [row for row in rows if row["spectral_strength"] == strength]
        aggregate[str(strength)] = {key: round(float(np.mean([row[key] for row in selected])), 6) for key in keys}
        for resolution in ("short", "medium", "long"):
            aggregate[str(strength)][resolution] = {
                key: round(float(np.mean([row["multi_resolution"][resolution][key] for row in selected])), 6)
                for key in ("spectral_flatness_mean", "spectral_instability_p95", "noise_floor_modulation_db_std")
            }
    baseline = aggregate["0.5"]
    candidate = aggregate["stationary_gate"]
    selected = (
        candidate["pitch_mae_cents"] <= baseline["pitch_mae_cents"] + 2
        and candidate["voicing_accuracy"] >= baseline["voicing_accuracy"] - .01
        and candidate["harmonic_to_noise_proxy_db"] >= baseline["harmonic_to_noise_proxy_db"] + .5
        and candidate["sample_jump_p999"] <= .8 * baseline["sample_jump_p999"]
        and all(
            candidate[resolution]["spectral_instability_p95"]
            <= .95 * baseline[resolution]["spectral_instability_p95"]
            for resolution in ("short", "medium", "long")
        )
    )
    report = {
        "status": "objective_candidate_human_pending" if selected else "diagnosed_no_fix_selected",
        "selection": "stationary_gate" if selected else None,
        "cases": len({row["case"] for row in rows}), "rows": rows,
        "aggregate_by_strength": dict(aggregate),
        "gate": {
            "pitch_regression_max_cents": 2, "voicing_regression_max": .01,
            "minimum_hnr_gain_db": .5, "minimum_sample_jump_reduction": .2,
            "minimum_instability_reduction_each_resolution": .05,
            "human_listening_required": True,
        },
    }
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "aggregate_by_strength": report["aggregate_by_strength"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
