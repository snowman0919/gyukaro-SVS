import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from gyu_singer.paths import project_roots
from gyu_singer.research import validate_evidence_manifest, validate_status_registry


ROOT = Path(__file__).resolve().parents[1]


def test_project_roots_are_portable_and_environment_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv("GYUKARO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("GYUKARO_DATA_ROOT", str(tmp_path / "private-data"))
    monkeypatch.setenv("GYUKARO_CACHE_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("GYUKARO_EVIDENCE_ROOT", str(tmp_path / "evidence"))

    roots = project_roots()

    assert roots.project == (tmp_path / "repo").resolve()
    assert roots.data == (tmp_path / "private-data").resolve()
    assert roots.cache == (tmp_path / "models").resolve()
    assert roots.evidence == (tmp_path / "evidence").resolve()


def test_status_registry_separates_versions_and_preserves_rejections():
    registry = json.loads((ROOT / "configs/project_status.json").read_text())

    validate_status_registry(registry)
    models = {row["identifier"]: row for row in registry["models"]}
    assert registry["production_approved_model"] is None
    assert models["gtsinger-ja-soprano"]["status"] == "foundation_only"
    assert models["gtsinger-ko-soprano"]["status"] == "foundation_ko_lexical_unverified"
    assert models["gtsinger-ja-tenor"]["status"] == "rejected"
    assert models["gtsinger-gyu-mix20"]["status"] == "rejected"
    assert models["rc8-candidate-3"]["status"] == "rejected"
    assert models["truncated-soulx-k2"]["status"] == "rejected"
    assert models["truncated-soulx-k4"]["status"] == "rejected"
    assert all({"software_version", "model_family", "experiment_id", "release_status"} <= row.keys()
               for row in models.values())


def test_status_registry_rejects_release_status_drift():
    registry = json.loads((ROOT / "configs/project_status.json").read_text())
    registry["models"][0]["release_status"] = "production_approved"

    with pytest.raises(ValueError, match="release status"):
        validate_status_registry(registry)


def test_evidence_manifest_hashes_tracked_reports():
    manifest = json.loads((ROOT / "configs/research_evidence.json").read_text())

    validate_evidence_manifest(manifest, ROOT)
    for row in manifest["evidence"]:
        if not row["local_only"]:
            path = ROOT / row["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_repository_claims_and_dependency_boundary_match_evidence():
    readme = (ROOT / "README.md").read_text()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    ignore = (ROOT / ".gitignore").read_text()

    assert "No production-approved singing model currently exists." in readme
    assert "OpenUtau and release paths remain blocked." in readme
    assert project["requires-python"] == ">=3.11,<3.12"
    assert {"evaluation", "diffsinger", "training", "openutau", "factory", "dev"} <= set(
        project["optional-dependencies"]
    )
    assert "artifacts/**/*.wav" in ignore
    assert "*.ckpt" in ignore


def test_pure_research_modules_do_not_import_model_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import gyu_singer.paths, gyu_singer.research; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0


def test_touched_diagnostic_script_has_no_machine_specific_home_path():
    script = (ROOT / "scripts/run_gtsinger_gyu_identity_diagnostic.py").read_text()
    assert "/home/kotori9" not in script


def test_research_registry_check_cli():
    result = subprocess.run(
        [sys.executable, "scripts/check_research_evidence.py"],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS models=9 evidence=8" in result.stdout
