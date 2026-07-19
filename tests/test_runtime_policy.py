import json
from importlib import resources
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from gyu_singer.experiments.identity_adapters import IdentityFiLM, LowRankIdentityResidual
from gyu_singer.experiments.identity_protocol import identity_training_authorized, validate_identity_protocol
from gyu_singer.runtime_policy import (
    IMPLEMENTED_BACKENDS,
    RuntimePolicyError,
    load_backend_registry,
    resolve_backend,
    validate_backend_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registry_and_cli_backend_sets_do_not_drift():
    registry = load_backend_registry(ROOT / "configs/backend_registry.json")
    expected = {
        "hybrid-svs", "hybrid-soulx-phrase", "orchestration-v0.4",
        "gyu-singer-v0.5", "gyu-singer-v0.6", "gyu-singer-v0.7",
        "gyu-singer-v0.8", "gyu-singer-rc5", "gyu-singer-rc6",
        "gyu-singer-rc8", "gyu-singer-rc9", "hybrid-compact-experimental",
        "loop", "neural-vocalizer-baseline",
    }

    assert set(IMPLEMENTED_BACKENDS) == expected
    assert set(registry["backends"]) == expected
    validate_backend_registry(
        registry,
        json.loads((ROOT / "configs/project_status.json").read_text()),
        expected,
    )


def test_no_default_render_when_no_production_model_is_approved():
    registry = load_backend_registry(ROOT / "configs/backend_registry.json")

    with pytest.raises(RuntimePolicyError, match="no production-approved"):
        resolve_backend(None, False, registry)


@pytest.mark.parametrize("backend", ["gyu-singer-rc8", "gyu-singer-rc9"])
def test_rejected_and_blocked_backends_require_explicit_override(backend):
    registry = load_backend_registry(ROOT / "configs/backend_registry.json")

    with pytest.raises(RuntimePolicyError, match="--allow-experimental"):
        resolve_backend(backend, False, registry)

    decision = resolve_backend(backend, True, registry)
    assert decision.backend == backend
    assert decision.experimental_override is True
    assert decision.status in {"rejected", "blocked"}


def test_cli_frontend_does_not_require_a_runtime_backend():
    result = subprocess.run(
        [sys.executable, "-m", "gyu_singer.cli", "frontend", "--language", "ko", "가"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_render_fails_closed_and_override_is_audited(tmp_path):
    denied = subprocess.run(
        [sys.executable, "-m", "gyu_singer.cli", "render", "missing.json", "--output", str(tmp_path / "x.wav")],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert denied.returncode == 2
    assert "no production-approved" in denied.stderr

    allowed = subprocess.run(
        [sys.executable, "-m", "gyu_singer.cli", "--backend", "gyu-singer-rc8", "--allow-experimental", "render", "missing.json", "--output", str(tmp_path / "x.wav")],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert "EXPERIMENTAL_OVERRIDE" in allowed.stderr
    assert '"status": "rejected"' in allowed.stderr


def test_registry_forbids_package_and_openutau_for_nonproduction_backends():
    registry = load_backend_registry(ROOT / "configs/backend_registry.json")

    assert all(
        not row["package_allowed"] and not row["openutau_allowed"]
        for row in registry["backends"].values()
        if row["status"] != "production_approved"
    )


def test_identity_protocol_is_bounded_and_current_foundation_blocks_training():
    protocol = json.loads((ROOT / "configs/gtsinger_gyu_identity_candidates.json").read_text())
    status = json.loads((ROOT / "configs/project_status.json").read_text())

    validate_identity_protocol(protocol)
    assert [row["type"] for row in protocol["candidates"]] == [
        "fixed_gyu_speaker_embedding", "small_film", "low_rank_residual", "vocoder_conditioning",
    ]
    assert protocol["candidates"][-1]["enabled"] is False
    assert identity_training_authorized(protocol, status) is False
    assert protocol["optimizer_steps"] == 0


@pytest.mark.parametrize("adapter", [IdentityFiLM(8, 4), LowRankIdentityResidual(8, 4, rank=2)])
def test_bounded_identity_adapters_are_zero_equivalent_at_initialization(adapter):
    hidden = torch.randn(2, 5, 8)
    identity = torch.randn(2, 4)

    output = adapter(hidden, identity)

    assert torch.equal(output, hidden)
    assert 0 < sum(parameter.numel() for parameter in adapter.parameters()) < 1_000


def test_runtime_safety_report_is_reproducible():
    result = subprocess.run(
        [sys.executable, "scripts/check_runtime_safety.py", "--check"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS production=none identity_training=false" in result.stdout


def test_packaged_backend_policy_resource_matches_authoritative_config():
    packaged = resources.files("gyu_singer").joinpath("backend_registry.json").read_bytes()
    assert packaged == (ROOT / "configs/backend_registry.json").read_bytes()
