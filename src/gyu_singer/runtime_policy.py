from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path

from .paths import project_roots


IMPLEMENTED_BACKENDS = (
    "hybrid-svs", "hybrid-soulx-phrase", "orchestration-v0.4",
    "gyu-singer-v0.5", "gyu-singer-v0.6", "gyu-singer-v0.7", "gyu-singer-v0.8",
    "gyu-singer-rc5", "gyu-singer-rc6", "gyu-singer-rc8", "gyu-singer-rc9",
    "hybrid-compact-experimental", "loop", "neural-vocalizer-baseline",
)


class RuntimePolicyError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeDecision:
    backend: str
    status: str
    reason: str
    experimental_override: bool


def load_backend_registry(path: Path | None = None) -> dict:
    source = path or project_roots().project / "configs/backend_registry.json"
    if not source.is_file():
        source = resources.files("gyu_singer").joinpath("backend_registry.json")
    registry = json.loads(source.read_text())
    project_path = source.parent / "project_status.json"
    project_status = json.loads(project_path.read_text()) if project_path.is_file() else None
    validate_backend_registry(registry, project_status, set(IMPLEMENTED_BACKENDS))
    return registry


def validate_backend_registry(registry: dict, project_status: dict | None, cli_backends: set[str] | None = None) -> None:
    allowed = set(registry.get("allowed_statuses", []))
    backends = registry.get("backends", {})
    if not backends or not allowed:
        raise ValueError("backend registry requires statuses and backends")
    if cli_backends is not None and set(backends) != set(cli_backends):
        raise ValueError("CLI and backend registry drift")
    models = {row["identifier"]: row for row in project_status.get("models", [])} if project_status else None
    required = {"status", "project_model", "reason", "package_allowed", "openutau_allowed"}
    for name, row in backends.items():
        if not required <= row.keys() or row["status"] not in allowed:
            raise ValueError(f"invalid backend row: {name}")
        model = row["project_model"]
        if model is not None and models is not None:
            if model not in models or models[model]["status"] != row["status"]:
                raise ValueError(f"backend/project status drift: {name}")
        if row["status"] != "production_approved" and (row["package_allowed"] or row["openutau_allowed"]):
            raise ValueError(f"nonproduction backend exposed to release path: {name}")
    approved = [name for name, row in backends.items() if row["status"] == "production_approved"]
    expected = approved[0] if len(approved) == 1 else None
    if registry.get("production_approved_backend") != expected:
        raise ValueError("production backend pointer drift")


def resolve_backend(backend: str | None, allow_experimental: bool, registry: dict) -> RuntimeDecision:
    selected = backend or registry.get("production_approved_backend")
    if selected is None:
        raise RuntimePolicyError("no production-approved backend exists; choose an explicit backend with --allow-experimental for diagnostic use")
    try:
        row = registry["backends"][selected]
    except KeyError as error:
        raise RuntimePolicyError(f"unknown backend: {selected}") from error
    if row["status"] != "production_approved" and not allow_experimental:
        raise RuntimePolicyError(
            f"backend {selected!r} is {row['status']}; --allow-experimental is required: {row['reason']}"
        )
    return RuntimeDecision(selected, row["status"], row["reason"], row["status"] != "production_approved")
