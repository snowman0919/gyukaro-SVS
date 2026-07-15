#!/usr/bin/env python3
"""Verify real RC5 backend outputs against human-approved candidate4 bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    runtime = Path("artifacts/reports/rc5_runtime_smoke")
    approved = Path("artifacts/reports/rc5_stress_candidate4/listening")
    names = {"ko": "ko_neutral", "en": "en", "rapid_ko": "rapid_ko", "large_interval_ko": "large_interval_ko"}
    rows = [{"case": case, "runtime": str(runtime / f"{case}.wav"), "approved": str(approved / f"{target}.wav"), "sha256": sha(runtime / f"{case}.wav"), "exact_match": sha(runtime / f"{case}.wav") == sha(approved / f"{target}.wav")} for case, target in names.items()]
    report = {"status": "pass" if all(row["exact_match"] for row in rows) else "fail", "backend": "gyu-singer-rc5", "comparison": "byte-for-byte against human-approved candidate4", "rows": rows}
    (runtime / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__":
    main()
