import json
from pathlib import Path
import subprocess
import sys

import pytest

from gyu_singer.release_gate import REQUIRED_GATES, ReleaseGateError, decide_release
from gyu_singer.openutau_packager import PackageError, build_openutau_package


ROOT = Path(__file__).resolve().parents[1]


def test_central_release_gate_requires_every_frozen_dimension():
    gate = json.loads((ROOT / "configs/release_gate.json").read_text())
    decision = decide_release(gate)

    assert set(gate["gates"]) == set(REQUIRED_GATES)
    assert decision.status == "release_blocked"
    assert "approved_foundation" in decision.failed_gates
    assert "approved_identity_candidate" in decision.failed_gates
    assert "human_approval_record" in decision.failed_gates


def test_missing_gate_and_whisper_only_lexical_evidence_are_rejected():
    gate = json.loads((ROOT / "configs/release_gate.json").read_text())
    del gate["gates"]["artifact_gate"]
    with pytest.raises(ReleaseGateError, match="missing release gates"):
        decide_release(gate)

    gate = json.loads((ROOT / "configs/release_gate.json").read_text())
    gate["gates"]["phone_centered_lexical_gate"] = {
        "passed": True, "method": "whisper_only", "evidence": ["transcript.json"]
    }
    with pytest.raises(ReleaseGateError, match="Whisper cannot be the sole"):
        decide_release(gate)


@pytest.mark.parametrize("backend", ["gyu-singer-rc8", "gyu-singer-rc9"])
def test_rejected_and_blocked_assets_cannot_create_release_package(tmp_path, backend):
    with pytest.raises(PackageError, match="release package refused"):
        build_openutau_package(ROOT, tmp_path / "voicebank", backend, diagnostic=False)
    assert not (tmp_path / "voicebank").exists()


def test_diagnostic_package_is_unmistakable_checkpoint_free_and_reproducible(tmp_path):
    output = tmp_path / "rc8-diagnostic"

    first = build_openutau_package(ROOT, output, "gyu-singer-rc8", diagnostic=True)
    first_sums = (output / "SHA256SUMS").read_text()
    second = build_openutau_package(ROOT, output, "gyu-singer-rc8", diagnostic=True)

    assert first.status == second.status == "diagnostic_package_not_a_release"
    assert (output / "PACKAGE.json").read_text().startswith('{\n  "banner": "NOT A RELEASE')
    assert "NOT A RELEASE" in (output / "README.md").read_text()
    assert json.loads((output / "checkpoint_references.json").read_text())["bundled"] is False
    assert list(output.rglob("*.ckpt")) == []
    assert list(output.rglob("*.pt")) == []
    assert (output / "SHA256SUMS").read_text() == first_sums


def test_diagnostic_mode_requires_diagnostic_output_name(tmp_path):
    with pytest.raises(PackageError, match="-diagnostic"):
        build_openutau_package(ROOT, tmp_path / "voicebank", "gyu-singer-rc8", diagnostic=True)


def test_release_gate_report_is_reproducible():
    result = subprocess.run(
        [sys.executable, "scripts/check_release_gate.py", "--check"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS release=blocked failed=11" in result.stdout
