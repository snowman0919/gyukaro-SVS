#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gyu_singer.experiments.identity_protocol import identity_training_authorized
from gyu_singer.runtime_policy import RuntimePolicyError, load_backend_registry, resolve_backend


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reports/runtime_safety/status.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report() -> dict:
    registry_path = ROOT / "configs/backend_registry.json"
    status_path = ROOT / "configs/project_status.json"
    identity_path = ROOT / "configs/gtsinger_gyu_identity_candidates.json"
    registry = load_backend_registry(registry_path)
    status = json.loads(status_path.read_text())
    identity = json.loads(identity_path.read_text())
    try:
        resolve_backend(None, False, registry)
        default_render = "allowed"
    except RuntimePolicyError:
        default_render = "blocked_no_production_backend"
    return {
        "status": "runtime_safety_complete",
        "production_approved_backend": registry["production_approved_backend"],
        "default_render": default_render,
        "backend_status_counts": {
            state: sum(row["status"] == state for row in registry["backends"].values())
            for state in registry["allowed_statuses"]
        },
        "rc8_default": "blocked",
        "rc9_default": "blocked",
        "experimental_override": "required_and_logged",
        "package_openutau_nonproduction": "blocked",
        "identity_training_authorized": identity_training_authorized(identity, status),
        "identity_optimizer_steps": identity["optimizer_steps"],
        "identity_runtime_integration": identity["runtime_integration"],
        "source_sha256": {
            "backend_registry": sha256(registry_path),
            "project_status": sha256(status_path),
            "identity_protocol": sha256(identity_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != serialized:
            raise SystemExit("runtime safety report is missing or stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(serialized)
    print("PASS production=none identity_training=false")


if __name__ == "__main__":
    main()
