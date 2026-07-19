from __future__ import annotations

import hashlib
from pathlib import Path


MODEL_FIELDS = {
    "identifier", "software_version", "model_family", "experiment_id", "status",
    "release_status", "allowed_uses", "blocked_uses", "reason", "evidence_paths",
    "commit", "checkpoint_hash", "human_approval_state",
}
MODEL_STATUSES = {
    "accepted_experimental_baseline", "foundation_only", "foundation_ko_lexical_unverified",
    "foundation_machine_inconclusive",
    "rejected", "blocked", "production_approved",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_status_registry(registry: dict) -> None:
    models = registry.get("models", [])
    if not models or len({row.get("identifier") for row in models}) != len(models):
        raise ValueError("status registry requires unique models")
    for row in models:
        if not MODEL_FIELDS <= row.keys() or row["status"] not in MODEL_STATUSES:
            raise ValueError(f"invalid model status row: {row.get('identifier')}")
        if row["release_status"] == "production_approved" and row["status"] != "production_approved":
            raise ValueError("release status cannot exceed model status")
    approved = [row["identifier"] for row in models if row["status"] == "production_approved"]
    if registry.get("production_approved_model") != (approved[0] if len(approved) == 1 else None):
        raise ValueError("production-approved model pointer drift")


def validate_evidence_manifest(manifest: dict, root: Path) -> None:
    required = {"path", "type", "status", "sha256", "originating_experiment", "commit", "local_only"}
    rows = manifest.get("evidence", [])
    if not rows or len({row.get("path") for row in rows}) != len(rows):
        raise ValueError("evidence manifest requires unique rows")
    for row in rows:
        if not required <= row.keys():
            raise ValueError("incomplete evidence row")
        path = root / row["path"]
        if not row["local_only"] and (not path.is_file() or sha256(path) != row["sha256"]):
            raise ValueError(f"evidence hash mismatch: {row['path']}")
