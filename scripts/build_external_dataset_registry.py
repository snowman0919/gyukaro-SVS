#!/usr/bin/env python3
"""Write the reviewed external-data license registry before any bulk download."""
from __future__ import annotations

import json
from pathlib import Path


DATASETS = [
    {
        "id": "libritts_r", "name": "LibriTTS-R", "domain": "speech", "languages": ["en"],
        "source": "https://www.openslr.org/141/", "license": "CC BY 4.0",
        "license_evidence": "https://www.openslr.org/141/", "allowed_uses": ["training", "evaluation", "commercial use with attribution"],
        "redistribution": "CC BY 4.0; raw subset is not bundled", "status": "selected",
        "selection": "dev_clean only; quality-filtered manifest, not full 585-hour corpus",
        "model_distribution": "allowed with attribution; checkpoint may be distributed",
    },
    {
        "id": "vocalset", "name": "VocalSet 1.2", "domain": "singing", "languages": ["en"],
        "source": "https://zenodo.org/records/1442513", "license": "CC BY 4.0",
        "license_evidence": "https://zenodo.org/records/1442513", "allowed_uses": ["training", "evaluation", "commercial use with attribution"],
        "redistribution": "CC BY 4.0; raw audio is not bundled", "status": "selected",
        "selection": "filtered scales, arpeggios, long tones, and excerpts; no blind full-corpus training",
        "model_distribution": "allowed with attribution; checkpoint may be distributed",
    },
    {
        "id": "emilia", "name": "Emilia original", "domain": "speech", "languages": ["en", "ja", "ko"],
        "source": "https://huggingface.co/datasets/amphion/Emilia-Dataset", "license": "CC BY-NC 4.0 plus gated terms",
        "license_evidence": "https://huggingface.co/datasets/amphion/Emilia-Dataset/blob/main/README.md", "allowed_uses": ["non-commercial research"],
        "redistribution": "access may be shared only with colleagues accepting gated terms; upstream does not own source copyright", "status": "excluded",
        "selection": "none", "model_distribution": "excluded from production checkpoint to avoid NC/copyright ambiguity",
    },
    {
        "id": "emilia_yodas", "name": "Emilia-YODAS", "domain": "speech", "languages": ["en", "ja", "ko"],
        "source": "https://huggingface.co/datasets/amphion/Emilia-Dataset", "license": "CC BY 4.0",
        "license_evidence": "https://huggingface.co/datasets/amphion/Emilia-Dataset/blob/main/README.md", "allowed_uses": ["training", "evaluation", "commercial use with attribution"],
        "redistribution": "CC BY 4.0; raw audio is not bundled", "status": "deferred",
        "selection": "none; 2.1 TB source and raw-origin risk make it unnecessary for the first bounded experiment",
        "model_distribution": "potentially allowed after per-item provenance review",
    },
    {
        "id": "jvs", "name": "JVS", "domain": "speech", "languages": ["ja"],
        "source": "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus", "license": "custom non-commercial research/personal-use terms",
        "license_evidence": "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus", "allowed_uses": ["academic research", "non-commercial research", "personal use"],
        "redistribution": "not permitted except a small public sample allowance", "status": "excluded",
        "selection": "none", "model_distribution": "excluded from production checkpoint without commercial permission",
    },
    {
        "id": "gtsinger", "name": "GTSinger", "domain": "singing", "languages": ["multi"],
        "source": "https://huggingface.co/datasets/GTSinger/GTSinger", "license": "CC BY-NC-SA 4.0 plus indemnity terms",
        "license_evidence": "https://huggingface.co/datasets/AaronZ345/GTSinger/blob/main/dataset_license.md", "allowed_uses": ["non-commercial training", "non-commercial evaluation"],
        "redistribution": "CC BY-NC-SA 4.0 and ShareAlike", "status": "excluded",
        "selection": "none", "model_distribution": "excluded so production checkpoint is not forced into NC-SA terms",
    },
    {
        "id": "jvs_music", "name": "JVS-MuSiC", "domain": "singing", "languages": ["ja"],
        "source": "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music", "license": "custom non-commercial research/personal-use terms",
        "license_evidence": "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music", "allowed_uses": ["academic research", "non-commercial research", "personal use"],
        "redistribution": "not permitted except a small public sample allowance", "status": "excluded",
        "selection": "none", "model_distribution": "excluded from production checkpoint without commercial permission",
    },
    {
        "id": "singnet", "name": "SingNet", "domain": "singing", "languages": ["multi"],
        "source": "https://arxiv.org/abs/2505.09325", "license": "no compatible released-data license verified",
        "license_evidence": "https://arxiv.org/abs/2505.09325", "allowed_uses": [], "redistribution": "unverified", "status": "excluded",
        "selection": "none", "model_distribution": "excluded until an official compatible data/model license is verified",
    },
]


def main() -> None:
    root = Path("data/external"); root.mkdir(parents=True, exist_ok=True)
    registry = {"schema": 1, "reviewed_on": "2026-07-15", "intended_use": "redistributable GYU Singer production checkpoint", "raw_audio_bundled": False, "datasets": DATASETS}
    (root / "dataset_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    rows = ["# External dataset licenses", "", "Raw external audio is never committed or bundled.", "", "| Dataset | License | Decision | Distribution |", "|---|---|---|---|"]
    rows += [f"| {d['name']} | {d['license']} | {d['status']}: {d['selection']} | {d['model_distribution']} |" for d in DATASETS]
    rows += ["", "Evidence URLs and exact use restrictions are preserved in `dataset_registry.json`."]
    (root / "DATASET_LICENSES.md").write_text("\n".join(rows) + "\n")
    selected = [d["name"] for d in DATASETS if d["status"] == "selected"]
    excluded = [d["name"] for d in DATASETS if d["status"] == "excluded"]
    report = "# Dataset and license audit\n\n" + f"Production training selected only {', '.join(selected)} (CC BY 4.0). " + f"Excluded from distributed weights: {', '.join(excluded)}. Emilia-YODAS remains deferred pending item-level provenance review.\n\n" + "No bulk download occurred before this registry was written. Selected raw files stay under ignored `data/external/raw/`; only reproducible manifests and quality metadata are committed.\n"
    Path("docs/dataset_and_license_audit.md").write_text(report)
    assert len(selected) == 2 and all(d["license"] == "CC BY 4.0" for d in DATASETS if d["status"] == "selected")
    print(json.dumps({"selected": selected, "excluded": excluded, "registry": str(root / "dataset_registry.json")}, indent=2))


if __name__ == "__main__":
    main()
