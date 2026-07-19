#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gyu_singer.release_gate import decide_release


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/release_gate.json"
OUTPUT = ROOT / "artifacts/reports/release_gate/status.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    gate = json.loads(SOURCE.read_text())
    decision = decide_release(gate)
    report = {
        "status": decision.status,
        "failed_gate_count": len(decision.failed_gates),
        "failed_gates": list(decision.failed_gates),
        "whisper_role": gate["whisper_role"],
        "release_package_created": False,
        "openutau_release_created": False,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != serialized:
            raise SystemExit("release gate report is missing or stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(serialized)
    print(f"PASS release=blocked failed={len(decision.failed_gates)}")


if __name__ == "__main__":
    main()
