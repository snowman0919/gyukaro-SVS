#!/usr/bin/env python3
"""Summarize score-native OpenUtau diagnostics without making a release claim."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts/reports/openutau_native_diffsinger"


def read(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def row(label: str, name: str, *, identity: bool = False) -> dict:
    report = read(name)
    quality = report["quality"]
    result = {
        "label": label,
        "audio_path": quality["audio_path"],
        "audio_sha256": quality["audio_sha256"],
        "automated_status": report["status"],
        "whisper_similarity": quality["asr_lyric_similarity"],
        "pitch_p90_cents": quality["pitch_p90_abs_cents"],
        "clip_fraction": quality["clip_fraction"],
        "waveform_sung_region": quality.get("waveform_sung_region"),
        "human_listening": "pending",
        "release_allowed": False,
    }
    if identity:
        result |= {
            "wavlm_to_gyu": quality["wavlm_to_identity_reference"],
            "ecapa_to_gyu": quality["ecapa_to_identity_reference"],
        }
    return result


def main() -> None:
    rows = [
        row("stock_depth_0_without_embedded_voicing", "song_20notes_source15000_stock_depth0_quality.json"),
        row("foundation_embedded_voicing", "song_20notes_source15000_final_smoke_quality.json", identity=True),
        *[
            row(f"speaker_embedding_{label}", f"song_20notes_identity_v2_{label}_stock_embedded_quality.json", identity=True)
            for label in ("g05", "g10", "g20")
        ],
        *[
            row(f"register_blend_{label}", f"song_20notes_register_blend_{label}_stock_embedded_quality.json", identity=True)
            for label in ("05", "10")
        ],
        row("bounded_mel_adapter_100", "song_20notes_gyu_mel_adapter_100_stock_quality.json", identity=True),
        *[
            row(f"gain_neutral_envelope_{label}", f"song_20notes_gyu_envelope_adapter_{label}_stock_quality.json", identity=True)
            for label in ("25", "50", "100")
        ],
    ]
    report = {
        "overall_status": "fail_no_release",
        "question_can_current_outputs_be_understood": "old v0.4/v0.6 no; current foundation diagnostic only has strong STT evidence",
        "primary_root_cause": "OpenUtau invoked an untrained stochastic diffusion path at depth 0.6",
        "secondary_root_cause": "stock OpenUtau passed score F0 through unvoiced consonant frames",
        "fixes_proven": [
            "force max_depth=0 for the auxiliary-decoder-only checkpoint",
            "embed a token-derived zero-F0 unvoiced mask inside the portable acoustic ONNX",
            "run the official DiffSinger renderer in an unmodified stock OpenUtau checkout",
            "require free Whisper, RMVPE F0 and waveform analysis for every reported WAV",
        ],
        "identity_conclusion": (
            "speaker embedding, low-register checkpoint interpolation, bounded neural mel residual, "
            "and gain-neutral spectral envelope did not improve both WavLM and ECAPA; none is a valid GYU voice"
        ),
        "best_technical_debug_baseline": "foundation_embedded_voicing",
        "best_debug_baseline_is_gyu": False,
        "package_status": "evaluation_only",
        "copyrighted_song_audio_packaged": False,
        "final_tag_created": False,
        "release_allowed": False,
        "human_listening_required": True,
        "rows": rows,
    }
    output = REPORTS / "reassessment.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    foundation = next(item for item in rows if item["label"] == "foundation_embedded_voicing")
    stock = next(item for item in rows if item["label"] == "stock_depth_0_without_embedded_voicing")
    document = f"""# OpenUtau DiffSinger quality reassessment

Overall status: **FAIL — no release**

The earlier v0.4/v0.6 outputs are not accepted as intelligible. The current work fixes two concrete score-native pipeline defects but does not yet produce a validated GYU voice.

## Proven root causes and fixes

1. The acoustic checkpoint trained only its deterministic auxiliary decoder, while the package requested diffusion depth 0.6. The stochastic path was untrained. The package now caps depth at 0.
2. Stock OpenUtau supplied nonzero score F0 through unvoiced consonants. A token-derived F0 mask is now embedded inside the acoustic ONNX, so no OpenUtau core fork is required.

On the same 20-note stock-OpenUtau phrase, Whisper similarity changed from `{stock['whisper_similarity']}` to `{foundation['whisper_similarity']}` and pitch p90 error from `{stock['pitch_p90_cents']}` to `{foundation['pitch_p90_cents']}` cents. Clipping remained `{foundation['clip_fraction']}`. The associated waveform/spectrogram/F0 review plot is generated beside the quality JSON.

## What failed

The foundation remains a GTSinger Japanese soprano voice. Four bounded identity attempts were rejected:

- learned speaker-embedding mixtures;
- low-register checkpoint interpolation;
- a 16,576-parameter phrase-paired neural mel residual;
- a gain-neutral 128-bin spectral-envelope direction.

None improved both WavLM and ECAPA against real GYU while preserving all other gates. Therefore none is enabled or described as GYU identity.

## Mandatory reporting rule

Every future audio candidate must include its SHA-256, free Whisper transcript/similarity, RMVPE pitch and voicing metrics, clipping, whole-file and sung-region waveform/spectral metrics, and a waveform/spectrogram/F0 plot. Human listening remains mandatory and cannot be replaced by these metrics.

No final tag or release is permitted from this state. Evaluation audio derived from the user-provided song is not packaged.
"""
    (ROOT / "docs/openutau_diffsinger_quality_reassessment.md").write_text(document)
    print(json.dumps({key: report[key] for key in (
        "overall_status", "primary_root_cause", "secondary_root_cause",
        "best_technical_debug_baseline", "best_debug_baseline_is_gyu", "release_allowed")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
