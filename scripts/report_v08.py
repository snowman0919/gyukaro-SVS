#!/usr/bin/env python3
"""Consolidate measured component selection into the v0.8 evidence report."""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf


def read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    prosody = read("artifacts/reports/independent_prosody_evaluation.json")
    v06 = read("artifacts/reports/v06_identity_style_ablation.json")["identity_ablation"]
    v07 = read("artifacts/reports/v07_identity_style_ablation.json")["identity_ablation"]
    style = read("artifacts/reports/v07_style_semantics.json")
    renders = {}
    for language in ("ko", "en", "ja"):
        path = Path(f"artifacts/reports/v08_quality_{language}.wav")
        info = sf.info(path)
        renders[language] = {"path": str(path), "sample_rate": info.samplerate, "channels": info.channels, "duration_sec": round(info.duration, 3)}
    report = {
        "production": {
            "backend": "gyu-singer-v0.8",
            "prosody": "v0.5 real-GYU controller",
            "identity": "v0.7 actual-SoulX-latent identity adapter",
            "style": "v0.7 actual-SoulX-latent style adapter",
            "decoder": "SoulX phrase decoder",
            "per_note_tts": False,
            "waveform_pitch_shift": False,
        },
        "prosody_selection": {
            "score_rows": prosody["rows"],
            "independent_from_target_f0": prosody["score_independent_from_target_f0"],
            "aggregates": {name: value["aggregate"] for name, value in prosody["runs"].items()},
            "selected": "v0.5_real_gyu_controller",
            "reason": "lowest pitch MAE and prior production evidence; v0.6 wins some transition metrics but is not consistent",
        },
        "identity_selection": {
            "v0.6_student_minus_none": v06["student_minus_no_identity"],
            "v0.7_student_minus_none": v07["student_minus_no_identity"],
            "v0.7_cross_language_consistency": v07["cross_language_identity_consistency"],
            "selected": "v0.7_real_latent",
            "reason": "actual decoder-latent training, positive mean WavLM/ECAPA deltas, and higher student cross-language consistency; confidence intervals cross zero, so quality gain remains modest",
        },
        "style_selection": {
            "selected": "v0.7_real_latent",
            "semantic_status": style["semantic_status"],
            "validated_directions": style["validated_semantic_directions_pass"],
            "limitation": "breathy and energetic proxies validate; soft, dark, and bright remain explicitly relabeled relative styles",
        },
        "renders": renders,
    }
    Path("artifacts/reports/evaluation_v08.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/evaluation_v0.8.md").write_text("# v0.8 production evaluation\n\n" + json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
