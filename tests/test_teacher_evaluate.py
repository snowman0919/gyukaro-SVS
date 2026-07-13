import importlib.util
from pathlib import Path


def test_teacher_evaluator_normalizes_language_aliases():
    spec = importlib.util.spec_from_file_location("teacher_evaluate", Path("scripts/evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.script_language_score("하늘에 빛이 내려와", "Korean") == 1.0
    assert module.levenshtein_similarity("하늘에 빛이 내려와.", "하늘에 빛이 내려와") == 1.0


def test_agreement_is_limited_to_the_same_benchmark_item():
    spec = importlib.util.spec_from_file_location("teacher_evaluate", Path("scripts/evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = [
        {"id": "item_1", "teacher": "fish"},
        {"id": "item_1", "teacher": "higgs"},
        {"id": "item_2", "teacher": "moss"},
    ]
    assert module.agreement_peers(rows[:2], rows[0]) == [rows[1]]
    assert module.passes_agreement(None)
    assert module.passes_agreement(0.5)
    assert not module.passes_agreement(0.49)
