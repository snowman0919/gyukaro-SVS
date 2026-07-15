#!/usr/bin/env python3
"""Verify RC6 candidate hashes plus resident deterministic/restart evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("artifacts/reports/rc6_runtime_smoke"); root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("artifacts/reports/rc6_backend_candidate/manifest.json").read_text())
    runtime = json.loads(Path("artifacts/reports/runtime_rc6_stress.json").read_text())
    rows = []
    for case, item in manifest["files"].items():
        path = Path(item["path"])
        rows.append({"case": case, "path": str(path), "expected_sha256": item["sha256"],
                     "actual_sha256": sha(path), "exact_match": path.is_file() and sha(path) == item["sha256"]})
    checks = {
        "candidate_hashes_match": all(row["exact_match"] for row in rows),
        "resident_backend_rc6": runtime.get("backend") == "gyu-singer-rc6",
        "resident_stress_pass": runtime.get("pass") is True,
        "deterministic_repeats": runtime.get("repeat_unique_sha256") == 1,
        "restart_stable": runtime.get("checks", {}).get("restart_stable") is True,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "backend": "gyu-singer-rc6",
        "method": "candidate file hashes plus actual resident repeat/restart stress evidence",
        "invalid_method_rejected": "fresh upstream generation is not compared byte-for-byte with separately generated upstream bytes",
        "checks": checks,
        "rows": rows,
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
