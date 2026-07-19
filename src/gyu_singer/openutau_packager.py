from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .release_gate import decide_release
from .runtime_policy import load_backend_registry


class PackageError(ValueError):
    pass


@dataclass(frozen=True)
class PackageResult:
    path: Path
    status: str


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_openutau_package(root: Path, output: Path, backend: str, diagnostic: bool = False) -> PackageResult:
    registry = load_backend_registry(root / "configs/backend_registry.json")
    if backend not in registry["backends"]:
        raise PackageError(f"unknown backend: {backend}")
    row = registry["backends"][backend]
    gate = json.loads((root / "configs/release_gate.json").read_text())
    decision = decide_release(gate)
    if diagnostic:
        if not output.name.endswith("-diagnostic"):
            raise PackageError("diagnostic package output name must end in -diagnostic")
        status = "diagnostic_package_not_a_release"
    elif (
        row["status"] != "production_approved"
        or not row["package_allowed"]
        or not row["openutau_allowed"]
        or decision.status != "release_approved"
    ):
        raise PackageError(f"release package refused for {backend}: status={row['status']} gates={decision.status}")
    else:
        status = "release_package"

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    banner = "NOT A RELEASE — DIAGNOSTIC PACKAGE ONLY" if diagnostic else "RELEASE PACKAGE"
    files = {
        "PACKAGE.json": {"banner": banner, "status": status, "backend": backend, "backend_status": row["status"], "source_commit": commit},
        "singer.json": {"name": "GYU diagnostic" if diagnostic else "GYU", "author": "GYU Singer research", "web": "", "version": "diagnostic" if diagnostic else "approved"},
        "phonemizer_mapping.json": {"languages": ["ko", "en", "ja"], "mapping_status": "research_only" if diagnostic else "approved"},
        "model_configuration.json": {"backend": backend, "activation": "disabled" if diagnostic else "production_approved_only"},
        "checkpoint_references.json": {"bundled": False, "references": [], "reason": "diagnostic packages never bundle experimental checkpoints" if diagnostic else "resolved from approved manifest"},
        "language_support.json": {"ko": "unverified", "en": "unverified", "ja": "foundation_only"} if diagnostic else {"source": "approved_evaluation"},
        "sample_configuration.json": {"sample_rate": 48000, "channels": 1, "samples_bundled": False},
        "license_provenance.json": {"validated": False if diagnostic else True, "source_audio_bundled": False, "external_datasets_bundled": False},
        "evaluation_summary.json": {"release_decision": decision.status, "failed_gates": list(decision.failed_gates), "whisper_role": gate["whisper_role"]},
    }
    for name, value in files.items():
        _write_json(output / name, value)
    (output / "README.md").write_text(
        f"# {banner}\n\nBackend: `{backend}`. This directory contains reproducible metadata only; no checkpoint, source recording, rendered WAV, or release approval is included.\n"
    )
    sums = [f"{_sha256(path)}  {path.name}" for path in sorted(output.iterdir()) if path.name != "SHA256SUMS"]
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return PackageResult(output, status)
