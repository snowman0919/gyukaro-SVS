#!/usr/bin/env python3
"""Record the bounded score-native GYU adaptation and RVC rejection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    evaluation = json.loads(
        (ROOT / "artifacts/reports/mlp_singer_korean_probe/evaluation.json").read_text()
    )
    weights = ROOT / "data/cache/rvc/assets/weights"
    report = {
        "status": "rvc_rejected_content_and_artifact_regression",
        "production_integrated": False,
        "human_listening_requested": False,
        "score_native_source": "neosapience/mlp-singer official pretrained checkpoint",
        "score_native_source_revision": "7f4621ca04ee5e35c0e0a80b1fed785a55a51891",
        "score_native_source_license": "MIT",
        "rvc_revision": "7ef19867780cf703841ebafb565a4e47d1ea86ff",
        "rvc_license": "MIT",
        "rvc_data": {
            "source_rows": 66,
            "source_duration_minutes": 23.371,
            "active_voice_minutes": 9.926,
            "segments": 518,
            "segment_duration_minutes": 19.52,
            "original_recordings_modified": False,
            "denoise_or_dereverb": False,
            "independent_verified_score_rows_used": 0,
        },
        "content_features": {
            "model": "facebook/hubert-base-ls960",
            "implementation": "Transformers HubertModel last hidden state",
            "dimension": 768,
            "caveat": "architecture-compatible with the RVC v2 expectation, not bit-identical to official fairseq extraction",
        },
        "training": {
            "sample_rate": 48000,
            "f0": True,
            "pitch_extractor": "official RMVPE",
            "batch_size": 16,
            "first_stage_epochs": 5,
            "continuation_epochs": 15,
            "continuation_generator_initialization": "first-stage epoch-5 generator",
            "continuation_discriminator_initialization": "official base discriminator; optimizer state reset",
            "label": "e5_plus15, not a continuous optimizer-state e20 run",
        },
        "checkpoints": {
            "e5": {
                "path": str((weights / "gyu_rvc_e5.pth").relative_to(ROOT)),
                "sha256": sha256(weights / "gyu_rvc_e5.pth"),
            },
            "e5_plus15": {
                "path": str((weights / "gyu_rvc_e5_plus15.pth").relative_to(ROOT)),
                "sha256": sha256(weights / "gyu_rvc_e5_plus15.pth"),
            },
        },
        "aggregate": {
            key: evaluation["aggregate"][key]
            for key in ("base_c6", "rvc_e5", "rvc_e5_plus15")
        },
        "acceptance": {
            "minimum_per_case_asr_similarity": 0.75,
            "identity_gain_required_on_both_wavlm_and_ecapa": 0.01,
            "hf_spike_must_beat_rc6": True,
            "selected": evaluation["selected"],
        },
        "finding": (
            "RVC raises GYU speaker similarity but destroys lyrics; the continuation also "
            "raises the high-frequency spike proxy. More training is rejected."
        ),
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/score_native_conversion_probe.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
