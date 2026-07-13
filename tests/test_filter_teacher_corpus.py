import importlib.util
from pathlib import Path


def test_filter_keeps_agreed_rows_and_caps_weak_speech_weight():
    spec = importlib.util.spec_from_file_location("teacher_filter", Path("scripts/filter_teacher_corpus.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = [
        {"id": "same_prompt", "language": "ko", "teacher": "fish", "quality_status": "teacher_gate_pass_unadmitted", "teacher_agreement_score": .8, "overall_confidence": .8},
        {"id": "same_prompt", "language": "ko", "teacher": "higgs", "quality_status": "teacher_gate_pass_unadmitted", "teacher_agreement_score": .6, "overall_confidence": .7},
        {"id": "same_prompt", "language": "ko", "teacher": "moss", "quality_status": "review_required", "teacher_agreement_score": .9, "overall_confidence": .9},
    ]
    selected = module.filter_rows(rows)
    assert [row["teacher"] for row in selected] == ["fish", "higgs"]
    assert selected[0]["teacher_role"] == "primary_representation_teacher"
    assert all(.05 <= row["trust_weight"] <= .20 for row in selected)
    assert {row["training_use"] for row in selected} == {"representation_distillation_only_not_singing_decoder"}
