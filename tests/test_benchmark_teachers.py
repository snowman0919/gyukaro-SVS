import importlib.util
import json
from pathlib import Path


def test_benchmark_language_contract_and_audio_validation():
    spec = importlib.util.spec_from_file_location("benchmark_teachers", Path("scripts/benchmark_teachers.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.LANGUAGES == {"ko": "Korean", "en": "English", "ja": "Japanese"}
    assert module.valid_audio(Path("examples/smoke.json")) is False


def test_fish_payload_encodes_authorized_reference(tmp_path):
    spec = importlib.util.spec_from_file_location("benchmark_teachers", Path("scripts/benchmark_teachers.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"audio")
    payload = module.request_payload({"text": "hello", "reference_text": "ref", "reference_audio_path": str(reference)}, "fish", 42)
    assert payload == {"text": "hello", "references": [{"text": "ref", "audio": "YXVkaW8="}], "format": "wav", "max_new_tokens": 42, "use_memory_cache": "on"}


def test_manifest_replace_leaves_a_complete_jsonl(tmp_path):
    spec = importlib.util.spec_from_file_location("benchmark_teachers", Path("scripts/benchmark_teachers.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path / "teacher.jsonl"
    module.write_manifest(output, {"one": {"id": "one"}})
    assert [json.loads(line) for line in output.read_text().splitlines()] == [{"id": "one"}]
    assert not output.with_suffix(".jsonl.tmp").exists()
