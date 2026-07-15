#!/usr/bin/env python3
"""Render the concise external-data audit from registries and measurements."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    registry = json.loads(Path("data/external/dataset_registry.json").read_text())
    report = json.loads(Path("artifacts/reports/external_acoustic_quality.json").read_text())
    zeroth = json.loads(Path("artifacts/reports/diffsinger_zeroth_prior.json").read_text())
    rows = [json.loads(line) for line in Path(report["manifest"]).read_text().splitlines() if line]
    selected = [dataset for dataset in registry["datasets"] if dataset["status"] == "selected"]
    excluded = [dataset for dataset in registry["datasets"] if dataset["status"] == "excluded"]
    failures = Counter(gate for row in rows for gate, passed in row["quality_gates"].items() if not passed)
    accepted_hours = sum(row["duration_sec"] for row in rows if row["accepted"]) / 3600
    lines = [
        "# Dataset and license audit", "",
        "## License decision", "",
        f"Production experiments select only {', '.join(item['name'] for item in selected)} (CC BY 4.0).",
        f"Excluded from distributed weights: {', '.join(item['name'] for item in excluded)}. Emilia-YODAS is deferred pending item-level provenance review.",
        "Raw external audio and the repaired local VocalSet archive are ignored and never bundled.", "",
        "## Bounded quality-filtered subset", "",
        f"The reproducible selector considered {report['rows']} files and accepted {report['accepted']} ({accepted_hours:.3f} hours): "
        + ", ".join(f"{name} {values['accepted']}/{values['rows']}" for name, values in report["by_dataset"].items()) + ".",
        f"Accepted split counts are {report['by_split']}; speakers are disjoint across splits: {report['speaker_disjoint_splits']}.",
        "Measured gates cover clipping, DC offset, level, SNR proxy, high-frequency energy, duration, WavLM speaker consistency, and Whisper text agreement for LibriTTS-R.",
        "VocalSet rows are isolated unaccompanied vocals; non-lexical technique clips have no fabricated ASR score. Music-background evidence is therefore source provenance, not a learned classifier.",
        f"Rejected-gate counts: {dict(sorted(failures.items()))}.", "",
        "The official VocalSet archive checksum matches Zenodo, but its ZIP offsets overflow. A local `zip -FF` recovery copy is used only to decode selected WAVs, each validated with libsndfile.", "",
        f"Zeroth-Korean contributes a separate bounded {zeroth['rows']}-utterance ({zeroth['hours']:.3f} hour), four-speaker Korean speech prior. MMS-CTC timings are explicitly inferred; these rows are neither singing nor GYU ground truth.", "",
        "Regenerate with `python scripts/build_external_dataset_registry.py`, `python scripts/prepare_external_acoustic_data.py`, then `python scripts/report_external_acoustic_data.py`.",
    ]
    Path("docs/dataset_and_license_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
