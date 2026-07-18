from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectRoots:
    project: Path
    data: Path
    cache: Path
    evidence: Path


def project_roots() -> ProjectRoots:
    project = Path(os.environ.get("GYUKARO_ROOT", Path(__file__).resolve().parents[2])).resolve()
    data = Path(os.environ.get("GYUKARO_DATA_ROOT", project / "data")).resolve()
    return ProjectRoots(
        project=project,
        data=data,
        cache=Path(os.environ.get("GYUKARO_CACHE_ROOT", data / "cache")).resolve(),
        evidence=Path(os.environ.get("GYUKARO_EVIDENCE_ROOT", project / "artifacts/reports")).resolve(),
    )
