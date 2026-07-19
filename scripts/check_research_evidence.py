#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from gyu_singer.research import validate_evidence_manifest, validate_status_registry


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    status = json.loads((ROOT / "configs/project_status.json").read_text())
    evidence = json.loads((ROOT / "configs/research_evidence.json").read_text())
    validate_status_registry(status)
    validate_evidence_manifest(evidence, ROOT)
    print(f'PASS models={len(status["models"])} evidence={len(evidence["evidence"])}')


if __name__ == "__main__":
    main()
