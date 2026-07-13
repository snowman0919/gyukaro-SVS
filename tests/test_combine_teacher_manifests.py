import importlib.util
from pathlib import Path


def test_combine_rejects_duplicate_teacher_item(tmp_path):
    spec = importlib.util.spec_from_file_location("combine_teacher_manifests", Path("scripts/combine_teacher_manifests.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = tmp_path / "teacher.jsonl"
    manifest.write_text('{"teacher":"fish","id":"item_1"}\n')
    assert module.combine([manifest]) == [{"teacher": "fish", "id": "item_1"}]
    try:
        module.combine([manifest, manifest])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate teacher item was accepted")
