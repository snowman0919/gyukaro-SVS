import importlib.util
from pathlib import Path


def test_benchmark_language_contract_and_audio_validation():
    spec = importlib.util.spec_from_file_location("benchmark_teachers", Path("scripts/benchmark_teachers.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.LANGUAGES == {"ko": "Korean", "en": "English", "ja": "Japanese"}
    assert module.valid_audio(Path("examples/smoke.json")) is False
