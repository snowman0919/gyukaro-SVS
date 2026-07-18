import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
from evaluate_korean_phone_reassessment import resample_audio  # noqa: E402

from gyu_singer.diffsinger.linguistic_adapter import (
    KoreanLinguisticAdapter,
    freeze_for_linguistic_adaptation,
)
from gyu_singer.evaluation.korean_lexical import (
    aggregate_phone_evidence,
    classify_alignment,
    edit_counts,
    korean_lexical_decision,
)
from gyu_singer.experiments.korean_phones import (
    encode_korean,
    mms_alignment_target,
    representation_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def test_three_korean_representations_are_deterministic_and_auditable():
    text = "가까카값"
    baseline = encode_korean(text, "ko_components_v1")
    canonical = encode_korean(text, "ko_canonical_v1")
    rhyme = encode_korean(text, "ko_onset_rhyme_v1")

    assert baseline == encode_korean(text, "ko_components_v1")
    assert canonical.audit_text == text
    assert rhyme.audit_text == text
    assert "ko_canonical_k_tense" in canonical.symbols
    assert "ko_canonical_k_aspirated" in canonical.symbols
    assert any(symbol.startswith("ko_rhyme_") for symbol in rhyme.symbols)
    assert baseline.symbols != canonical.symbols != rhyme.symbols


def test_probe_manifest_covers_required_korean_stress_and_phonology():
    manifest = json.loads((ROOT / "configs/korean_phone_probe.json").read_text())
    categories = {category for row in manifest["probes"] for category in row["categories"]}

    assert manifest["seeds"] == [7, 21, 42]
    assert {
        "single_syllable", "minimal_contrast", "repeated_syllable", "ordinary", "rapid",
        "sustain", "large_interval", "phrase_boundary", "coda", "tense", "aspirated",
        "liaison", "nasalization", "liquid_assimilation",
    } <= categories
    assert set(manifest["representations"]) == {
        "ko_components_v1", "ko_canonical_v1", "ko_onset_rhyme_v1"
    }


def test_representation_coverage_is_bounded():
    result = representation_coverage(["가", "값", "신라"], "ko_onset_rhyme_v1")
    assert result["observed_symbols"] <= result["maximum_symbols"] == 607
    assert result["unknown_characters"] == []


def test_mms_alignment_target_is_deterministic_roman_character_evidence():
    assert mms_alignment_target("가 까 카") == "gakkaka"


def test_phone_edit_counts_separate_insert_delete_substitute():
    assert edit_counts(list("abc"), list("axbc")) == {"insertions": 1, "deletions": 0, "substitutions": 0}
    assert edit_counts(list("abc"), list("ac")) == {"insertions": 0, "deletions": 1, "substitutions": 0}
    assert edit_counts(list("abc"), list("axc")) == {"insertions": 0, "deletions": 0, "substitutions": 1}


def test_whisper_cannot_change_primary_korean_decision():
    evidence = {
        "calibration_status": "calibrated",
        "acoustic_representations": 2,
        "phone_error_rate": 0.1,
        "alignment_confidence": 0.9,
        "boundary_deviation": 0.05,
        "duration_deviation": 0.05,
        "seed_content_consistency": 0.98,
        "catastrophic_content_collapse": False,
        "acoustic_gates_pass": True,
    }

    good = korean_lexical_decision(evidence, auxiliary_stt_observation="정확한 가사")
    bad = korean_lexical_decision(evidence, auxiliary_stt_observation="와우 와우 와우")

    assert good == bad == "foundation_candidate_human_pending"


def test_uncalibrated_phone_evidence_is_machine_inconclusive():
    evidence = {
        "calibration_status": "insufficient_separation",
        "acoustic_representations": 1,
        "catastrophic_content_collapse": False,
        "acoustic_gates_pass": True,
    }
    assert korean_lexical_decision(evidence, "anything") == "foundation_machine_inconclusive"


def test_phone_aggregation_reports_seed_instability():
    rows = [
        {"case": "ordinary", "seed": 7, "phone_error_rate": 0.1, "content_embedding": [1.0, 0.0]},
        {"case": "ordinary", "seed": 21, "phone_error_rate": 0.2, "content_embedding": [1.0, 0.0]},
        {"case": "ordinary", "seed": 42, "phone_error_rate": 0.9, "content_embedding": [0.0, 1.0]},
    ]
    result = aggregate_phone_evidence(rows)
    assert result["phone_error_rate"]["maximum"] == pytest.approx(0.9)
    assert result["seed_content_consistency"] < 0.5


@pytest.mark.parametrize(
    ("source", "confidence", "expected"),
    [
        ("manual", 1.0, "manual"),
        ("forced", 0.9, "forced_aligned_high_confidence"),
        ("forced", 0.5, "forced_aligned_low_confidence"),
        ("inferred", 0.9, "inferred_only"),
        ("forced", 0.1, "rejected"),
    ],
)
def test_alignment_confidence_classification(source, confidence, expected):
    assert classify_alignment(source, confidence) == expected


def test_linguistic_adapter_starts_equivalent_and_is_only_trainable_surface():
    foundation = torch.nn.Linear(4, 4)
    adapter = KoreanLinguisticAdapter(vocabulary_size=8, hidden_size=3, foundation_size=4)
    freeze_for_linguistic_adaptation(foundation, adapter)
    base = torch.randn(2, 4)
    tokens = torch.tensor([1, 2])

    output = adapter(base, tokens)
    assert torch.equal(output, base)
    output.sum().backward()
    assert all(parameter.grad is None for parameter in foundation.parameters())
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in adapter.parameters())
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_phone_reassessment_cli_dry_run_does_not_load_models():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_korean_phone_reassessment.py", "--dry-run"],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary == {"calibration_references": 5, "candidate_outputs": 15, "representations": 3}


def test_phone_reassessment_resamples_to_model_rate():
    import numpy as np

    values = resample_audio(np.ones(44_100, dtype=np.float32), 44_100, 16_000)
    assert values.dtype == np.float32
    assert len(values) == 16_000


def test_alignment_audit_marks_every_boundary_source_without_manual_claims():
    rows = [json.loads(line) for line in (
        ROOT / "data/manifests/korean_alignment_audit.jsonl"
    ).read_text().splitlines() if line]
    assert len(rows) == 149
    assert all(row["classification"] in {
        "manual", "forced_aligned_high_confidence", "forced_aligned_low_confidence",
        "inferred_only", "rejected",
    } for row in rows)
    assert all(row["classification"] == "inferred_only" for row in rows)
    assert all(row["training_weight"] < 1.0 for row in rows)


def test_linguistic_adapter_protocol_is_frozen_but_training_is_blocked():
    protocol = json.loads((ROOT / "configs/korean_linguistic_adapter.json").read_text())
    assert protocol["selected_representation"] is None
    assert protocol["training_status"] == "blocked_no_selected_representation"
    assert protocol["identity_objective_used"] is False
    assert protocol["trainable_modules"] == ["korean_phone_embeddings", "linguistic_projection_adapter"]
    assert protocol["maximum_steps"] == 200


def test_inconclusive_foundation_has_blind_review_package_without_stt_hints():
    manifest = json.loads((
        ROOT / "artifacts/reports/korean_phone_reassessment/human_review_manifest.json"
    ).read_text())
    assert manifest["status"] == "foundation_machine_inconclusive"
    assert manifest["promotion_allowed"] is False
    assert len(manifest["items"]) == 5
    assert set(manifest["rubric"]) == {
        "phoneme_identity", "syllable_omission", "syllable_insertion", "consonant_clarity",
        "vowel_correctness", "timing", "pitch", "voicing", "artifacts", "naturalness",
    }
    assert all(set(item["blind_audio"]) == {"A", "B"} for item in manifest["items"])
    assert all("whisper" not in json.dumps(item).lower() for item in manifest["items"])
