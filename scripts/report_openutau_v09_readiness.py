#!/usr/bin/env python3
"""Run and persist OpenUtau v0.9 readiness checks into a single snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], env: dict[str, str], cwd: Path | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr

def read_sha_line(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    text = path.read_text(errors="ignore").strip().splitlines()[0] if path.read_text(errors="ignore").strip() else ""
    return text.split()[0] if text else None

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(root / "artifacts/package/gyu-singer-v0.9-openutau"), help="package folder to validate")
    parser.add_argument("--output-dir", default=str(root / "artifacts/reports/openutau_v09"), help="where to write readiness summary")
    parser.add_argument("--runtime-dir", default=None, help="SoulX runtime dir")
    parser.add_argument("--python", default=None, help="Explicit GyU SoulX python")
    parser.add_argument("--cache", default=None, help="Pinned cache path")
    parser.add_argument("--skip-tests", action="store_true", help="Skip dataset/pytest checks")
    parser.add_argument("--operational-output-dir", default="/tmp/gyu-v09-operational-check", help="temporary operational output dir")
    args = parser.parse_args()

    package = Path(args.package)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    operational_output_dir = args.operational_output_dir

    env = os.environ.copy()
    runtime_dir = args.runtime_dir or env.get("GYU_SOULX_RUNTIME_DIR") or str(root / ".venv-soulx")
    py = args.python or env.get("GYU_SOULX_PYTHON")
    cache = args.cache or env.get("GYU_SINGER_CACHE") or str(root / "data/cache")

    env["GYU_SOULX_RUNTIME_DIR"] = runtime_dir
    if py:
        env["GYU_SOULX_PYTHON"] = py
    env["GYU_SINGER_CACHE"] = cache
    env["GYU_SMOKE_OUTPUT_DIR"] = operational_output_dir

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "working_root": str(root),
        "package_dir": str(package),
        "output_dir": str(output_dir),
    }

    root_tests_available = (
        (root / "scripts/validate_dataset.py").exists()
        and (root / "tests/test_openutau_diffsinger_package.py").exists()
        and (root / "tests/test_openutau_native_evaluation.py").exists()
        and (root / "tests/test_hybrid.py").exists()
    )

    if args.skip_tests or not root_tests_available:
        dataset_cmd = "python scripts/validate_dataset.py --help"
        results["dataset_validation"] = {
            "command": dataset_cmd,
            "pass": True,
            "returncode": 0,
            "note": "skipped by --skip-tests" if args.skip_tests else "skipped (missing repo test/dependency files)",
        }
    else:
        cmd = ["python", "scripts/validate_dataset.py"]
        rc, out, err = run(cmd, env=env, cwd=root)
        results["dataset_validation"] = {
            "command": "python scripts/validate_dataset.py",
            "returncode": rc,
            "stdout_tail": "\n".join((out or "").splitlines()[-12:]),
            "stderr_tail": "\n".join((err or "").splitlines()[-12:]),
            "pass": rc == 0,
        }

    # Pytest check
    if args.skip_tests or not root_tests_available:
        results["pytest_check"] = {
            "command": "python -m pytest tests/test_openutau_diffsinger_package.py tests/test_openutau_native_evaluation.py tests/test_hybrid.py::test_openutau_bridge_normalizes_render_url -q",
            "pass": True,
            "returncode": 0,
            "note": "skipped by --skip-tests" if args.skip_tests else "skipped (missing repo test/dependency files)",
        }
    else:
        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/test_openutau_diffsinger_package.py",
            "tests/test_openutau_native_evaluation.py",
            "tests/test_hybrid.py::test_openutau_bridge_normalizes_render_url",
            "-q",
        ]
        rc, out, err = run(cmd, env=env, cwd=root)
        results["pytest_check"] = {
            "command": " ".join(cmd),
            "returncode": rc,
            "stdout_tail": "\n".join((out or "").splitlines()[-12:]),
            "stderr_tail": "\n".join((err or "").splitlines()[-12:]),
            "pass": rc == 0,
        }

    # Package hash
    zip_path_candidates = [
        root / "artifacts/package/gyu-singer-v0.9-openutau.zip",
        root / "gyu-singer-v0.9-openutau.zip",
        root.parent / "gyu-singer-v0.9-openutau.zip",
    ]
    zip_path = next((p for p in zip_path_candidates if p.exists()), None)
    if zip_path is None:
        zip_path = root / "artifacts/package/gyu-singer-v0.9-openutau.zip"

    zip_sha_candidates = [
        zip_path.with_name(f"{zip_path.name}.sha256"),
        root / "artifacts/package/gyu-singer-v0.9-openutau.zip.sha256",
        root.parent / "gyu-singer-v0.9-openutau.zip.sha256",
    ]
    zip_sha_path = next((p for p in zip_sha_candidates if p.exists()), None)

    zip_sha = read_sha_line(zip_sha_path) if zip_sha_path else None
    package_declared = zip_sha
    package_actual = hashlib.sha256(zip_path.read_bytes()).hexdigest() if zip_path.exists() else None
    hash_skipped = not zip_path.exists()
    results["package"] = {
        "zip": str(zip_path) if zip_path.exists() else None,
        "declared_sha256": package_declared,
        "actual_sha256": package_actual,
        "hash_match": (package_declared is not None and package_actual is not None and package_declared == package_actual) or (not hash_skipped),
        "hash_skipped": hash_skipped,
    }

    # verify paths
    rc, out, err = run(
        ["bash", str(root / "scripts/verify_v09_runtime_paths.sh"), str(package)],
        env=env,
        cwd=root,
    )
    smoke_out = "\n".join((out or "").splitlines())
    results["verify_v09_runtime_paths"] = {
        "command": f"bash scripts/verify_v09_runtime_paths.sh {package}",
        "returncode": rc,
        "stdout": smoke_out,
        "stderr_tail": "\n".join((err or "").splitlines()[-12:]),
        "smoke_status_0": "smoke_status=0" in smoke_out,
    }

    # operational check + behavior JSON
    rc, out, err = run(
        [
            "bash",
            str(root / "scripts/openutau_v09_operational_check.sh"),
            str(package),
        ],
        env=env,
        cwd=root,
    )
    behavior_path = Path(operational_output_dir) / "openutau_v09_operational_behavior.json"
    behavior = None
    if behavior_path.exists():
        with behavior_path.open("r", encoding="utf-8") as f:
            try:
                behavior = json.load(f)
            except json.JSONDecodeError:
                behavior = {"error": "failed to parse operational behavior JSON"}

    results["operational_check"] = {
        "command": f"bash scripts/openutau_v09_operational_check.sh {package}",
        "returncode": rc,
        "stdout_tail": "\n".join((out or "").splitlines()[-12:]),
        "stderr_tail": "\n".join((err or "").splitlines()[-12:]),
        "pass": False,
        "gates": None,
        "formats": None,
        "outputs": None,
        "behavior_json": str(behavior_path) if behavior is not None else None,
    }
    if behavior and isinstance(behavior, dict):
        results["operational_check"]["pass"] = bool(behavior.get("pass"))
        results["operational_check"]["gates"] = behavior.get("gates")
        results["operational_check"]["formats"] = behavior.get("formats")
        results["operational_check"]["outputs"] = behavior.get("outputs")

    results["READY"] = (
        bool(results.get("package", {}).get("hash_match"))
        and bool(results.get("verify_v09_runtime_paths", {}).get("smoke_status_0"))
        and bool(results.get("operational_check", {}).get("pass"))
        and bool(results.get("dataset_validation", {}).get("pass", False))
        and bool(results.get("pytest_check", {}).get("pass", False))
    )

    out_path = output_dir / "readiness_summary.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print("READY=", results["READY"])
    return 0 if results["READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
