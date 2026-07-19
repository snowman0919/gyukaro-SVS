#!/usr/bin/env python3
"""Calibrate phone-centered evidence without promoting Korean Whisper to a gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from gyu_singer.evaluation.korean_lexical import aggregate_phone_evidence, korean_lexical_decision
from gyu_singer.experiments.korean_phones import (
    REPRESENTATIONS,
    mms_alignment_target,
    representation_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_IDS = (212, 215, 216, 219, 220)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def resample_audio(values, source_rate: int, target_rate: int):
    import numpy as np
    from scipy.signal import resample_poly

    if source_rate == target_rate:
        return np.asarray(values, dtype=np.float32)
    return resample_poly(values, target_rate, source_rate).astype(np.float32)


def _score_boundaries(case: dict) -> list[float] | None:
    score = json.loads((ROOT / case["score_path"]).read_text())
    notes = score["notes"]
    syllables = [char for char in case["expected_lyrics"] if "가" <= char <= "힣"]
    if len(notes) != len(syllables):
        return None
    total = max(float(note["start"]) + float(note["duration"]) for note in notes)
    return [(float(note["start"]) + float(note["duration"])) / total for note in notes]


def run(output: Path) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoFeatureExtractor, HubertModel

    protocol = json.loads((ROOT / "data/manifests/gtsinger_gyu_identity_protocol.json").read_text())
    previous = json.loads((ROOT / "artifacts/reports/gtsinger_gyu_identity_diagnostic/foundation_ko_evaluation.json").read_text())
    cases = {row["id"]: row for row in protocol["cases"]}
    prior_rows = {(row["case"], row["seed"]): row for row in previous["rows"]}
    segments = {int(row["source_index"]): row for row in _jsonl(ROOT / "data/manifests/real_segments.jsonl")
                if int(row["source_index"]) in REFERENCE_IDS}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels()
    dictionary = {label: index for index, label in enumerate(labels)}
    mms = bundle.get_model().to(device).eval()

    def audio(path: Path) -> tuple[np.ndarray, int]:
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        return values.mean(1), rate

    def align(path: Path, text: str, expected_boundaries: list[float] | None = None) -> dict:
        values, rate = audio(path)
        values = resample_audio(values, rate, bundle.sample_rate)
        target = mms_alignment_target(text)
        tokens = torch.tensor([[dictionary[char] for char in target]], device=device)
        with torch.inference_mode():
            emission, _ = mms(torch.from_numpy(values)[None].to(device))
            log_probs = emission.log_softmax(-1)
            alignment, scores = torchaudio.functional.forced_align(log_probs, tokens)
        spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
        path_rows = [
            {"symbol": target[index], "start_frame": span.start, "end_frame": span.end,
             "mean_log_score": round(float(span.score), 6)}
            for index, span in enumerate(spans)
        ]
        token_scores = [row["mean_log_score"] for row in path_rows]
        result = {
            "target_roman_characters": target,
            "alignment_path": path_rows,
            "alignment_confidence": round(math.exp(statistics.fmean(token_scores)), 6),
            "mean_log_score": round(statistics.fmean(token_scores), 6),
            "path_coverage": round(sum(row["end_frame"] - row["start_frame"] for row in path_rows) / emission.shape[1], 6),
            "monotonic": all(left["end_frame"] <= right["start_frame"]
                             for left, right in zip(path_rows, path_rows[1:])),
        }
        if expected_boundaries and len(expected_boundaries) == len([c for c in text if "가" <= c <= "힣"]):
            ends, cursor = [], 0
            for char in text:
                if not "가" <= char <= "힣":
                    continue
                cursor += len(mms_alignment_target(char))
                ends.append(path_rows[cursor - 1]["end_frame"] / emission.shape[1])
            result["boundary_deviation"] = round(statistics.fmean(
                abs(actual - expected) for actual, expected in zip(ends, expected_boundaries)
            ), 6)
            expected_durations = [expected_boundaries[0]] + [right - left for left, right in zip(expected_boundaries, expected_boundaries[1:])]
            actual_durations = [ends[0]] + [right - left for left, right in zip(ends, ends[1:])]
            result["duration_deviation"] = round(statistics.fmean(
                abs(actual - expected) for actual, expected in zip(actual_durations, expected_durations)
            ), 6)
        else:
            result["boundary_deviation"] = None
            result["duration_deviation"] = None
        return result

    calibration = []
    reference_texts = [segments[index]["text"] for index in REFERENCE_IDS]
    for position, reference_id in enumerate(REFERENCE_IDS):
        path = ROOT / f"data/processed/master/{reference_id}.wav"
        matched = align(path, reference_texts[position])
        mismatched = align(path, reference_texts[(position + 1) % len(reference_texts)])
        calibration.append({
            "id": f"gyu_real_{reference_id}", "audio_path": str(path.relative_to(ROOT)),
            "audio_sha256": sha256(path), "script_source": "known_recording_script",
            "matched_mean_log_score": matched["mean_log_score"],
            "mismatched_mean_log_score": mismatched["mean_log_score"],
            "score_margin": round(matched["mean_log_score"] - mismatched["mean_log_score"], 6),
        })

    feature_extractor = AutoFeatureExtractor.from_pretrained(ROOT / "data/cache/hubert-base-ls960")
    hubert = HubertModel.from_pretrained(ROOT / "data/cache/hubert-base-ls960").to(device).eval()

    def content_embedding(path: Path) -> list[float]:
        values, rate = audio(path)
        values = resample_audio(values, rate, feature_extractor.sampling_rate)
        inputs = feature_extractor(values, sampling_rate=feature_extractor.sampling_rate,
                                   return_tensors="pt")
        with torch.inference_mode():
            hidden = hubert(inputs.input_values.to(device)).last_hidden_state.mean(1)[0]
        hidden = torch.nn.functional.normalize(hidden, dim=0)
        return hidden.cpu().tolist()

    rows = []
    for case_id, case in cases.items():
        boundaries = _score_boundaries(case)
        for seed in protocol["seeds"]:
            prior = prior_rows[(case_id, seed)]
            path = ROOT / prior["audio_path"]
            forced = align(path, case["expected_lyrics"], boundaries)
            rows.append({
                "case": case_id, "stress_category": case["stress_category"], "seed": seed,
                "representation": "ko_components_v1", "expected_lyrics": case["expected_lyrics"],
                "expected_phones": case["expected_phonemes"],
                "aligned_phones_or_posterior_path": forced["alignment_path"],
                "phone_insertions": None, "phone_deletions": None, "phone_substitutions": None,
                "phone_error_rate": None, "alignment_confidence": forced["alignment_confidence"],
                "alignment_mean_log_score": forced["mean_log_score"],
                "alignment_coverage": forced["path_coverage"], "alignment_monotonic": forced["monotonic"],
                "boundary_deviation": forced["boundary_deviation"],
                "duration_deviation": forced["duration_deviation"],
                "content_embedding": content_embedding(path),
                "auxiliary_stt_observation": prior["whisper_transcript"],
                "auxiliary_stt_lyric_similarity": prior["lyric_similarity"],
                "pitch_mae_cents": prior["pitch_mae_cents"],
                "pitch_p90_abs_cents": prior["pitch_p90_abs_cents"],
                "gross_pitch_error_rate": prior["gross_error_over_600_cents"],
                "voicing_accuracy": prior["voicing_accuracy"], "clip_fraction": prior["clip_fraction"],
                "hf_spike": prior["hf_spike_p99_over_median"], "sample_jump": prior["sample_jump_p999"],
                "waveform_discontinuity": prior["waveform_discontinuity"],
                "audio_path": prior["audio_path"], "audio_sha256": prior["audio_sha256"],
                "uncertainty_reason": "MMS_FA is target-conditioned alignment, not independent phone recognition",
            })

    # The two representations are complementary but neither independently recognizes Korean phones.
    comparable = [{**row, "phone_error_rate": 1.0 if row["phone_error_rate"] is None else row["phone_error_rate"]}
                  for row in rows]
    aggregate = aggregate_phone_evidence(comparable)
    for row in rows:
        row["seed_content_consistency"] = aggregate["seed_content_consistency"]
        row.pop("content_embedding")
        row["machine_lexical_decision"] = korean_lexical_decision({
            "calibration_status": "insufficient_phone_recognition",
            "acoustic_representations": 2,
            "catastrophic_content_collapse": False,
            "acoustic_gates_pass": True,
        }, row["auxiliary_stt_observation"])

    probe = json.loads((ROOT / "configs/korean_phone_probe.json").read_text())
    texts = [row["text"] for row in probe["probes"]]
    report = {
        "status": "foundation_machine_inconclusive",
        "primary_gate": "phone_centered",
        "whisper_role": "auxiliary_stt_observation",
        "whisper_primary_weight": 0.0,
        "calibration_status": "insufficient_phone_recognition",
        "calibration_reason": "MMS_FA supplies a target-conditioned path and HuBERT supplies content consistency, but no calibrated independent Korean singing phone recognizer is local.",
        "mms_model": str(bundle._path),
        "hubert_model": "data/cache/hubert-base-ls960",
        "calibration": calibration,
        "representations": [representation_coverage(texts, name) for name in REPRESENTATIONS],
        "rendered_representation": "ko_components_v1",
        "unrendered_representations": ["ko_canonical_v1", "ko_onset_rhyme_v1"],
        "representation_selection": None,
        "representation_selection_status": "lexical_machine_inconclusive",
        "aggregate": {
            "seed_content_consistency": aggregate["seed_content_consistency"],
            "phrase_seed_count": len(rows),
            "machine_inconclusive_count": sum(row["machine_lexical_decision"] == "foundation_machine_inconclusive" for row in rows),
        },
        "rows": rows,
        "linguistic_adapter": {
            "implementation": "KoreanLinguisticAdapter",
            "status": "not_started_no_selected_representation",
            "optimizer_steps": 0,
            "identity_objective_used": False,
        },
        "identity_training_allowed": False,
        "human_review_required": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(rows),
                      "seed_content_consistency": aggregate["seed_content_consistency"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/reports/korean_phone_reassessment/evaluation.json")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"calibration_references": 5, "candidate_outputs": 15,
                          "representations": len(REPRESENTATIONS)}))
        return
    run(args.output)


if __name__ == "__main__":
    main()
