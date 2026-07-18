#!/usr/bin/env python3
"""Build a source-qualified five-phrase GTSinger DiffSinger evaluation set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path[:0] = [str(ROOT / "scripts"), str(CACHE / "soulx-singer")]
from prepare_diffsinger_gtsinger_ja import (  # noqa: E402
    DATASET,
    normalized_phones,
    selected_rows,
)
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


INDICES = (165, 172, 174, 379, 380)


def build_ds_row(row: dict, audio_duration: float, f0: np.ndarray) -> dict:
    """Create one score-native row from manual timing and an RMVPE target grid."""
    phones = normalized_phones(row)
    durations = [float(value) for value in row["ph_durs"]]
    delta = audio_duration - sum(durations)
    if delta > .001:
        phones.append("SP")
        durations.append(delta)
    elif delta < -.001:
        durations[-1] += delta
    if abs(sum(durations) - audio_duration) >= .002:
        raise ValueError("manual phoneme duration does not match source audio")
    if durations[-1] <= 0:
        raise ValueError("manual phoneme duration became non-positive")
    if abs(len(f0) * .02 - audio_duration) > .04:
        raise ValueError("RMVPE grid does not match source audio")
    return {
        "offset": 0,
        "text": "".join(value for value in row["txt"] if not value.startswith("<")),
        "ph_seq": " ".join(phones),
        "ph_dur": " ".join(f"{value:.7f}" for value in durations),
        "f0_seq": " ".join(f"{value:.3f}" for value in f0),
        "f0_timestep": .02,
        "spk_mix": {"gts_ja_soprano": 1.0},
    }


def main() -> None:
    rows = selected_rows(json.loads((DATASET / "processed/Japanese/metadata.json").read_text()))
    output = ROOT / "data/external/work/gtsinger/heldout_eval"
    report_dir = ROOT / "artifacts/reports/diffsinger_gtsinger_heldout_set"
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda",
        target_sr=24_000,
        hop_size=480,
        verbose=False,
    )
    manifest = []
    for index in INDICES:
        identifier = f"gtsja{index:04d}"
        source = DATASET / rows[index]["wav_fn"]
        f0 = np.asarray(extractor.process(str(source), verbose=False), dtype=np.float32)
        ds = build_ds_row(rows[index], sf.info(source).duration, f0)
        ds_path = output / f"{identifier}.ds"
        ds_path.write_text(json.dumps([ds], ensure_ascii=False, indent=2) + "\n")
        manifest.append({
            "id": identifier,
            "expected_text": ds["text"],
            "source_audio_path": str(source.relative_to(ROOT)),
            "ds_path": str(ds_path.relative_to(ROOT)),
            "alignment_status": "dataset_provided_manual",
            "target_f0_source": "rmvpe",
            "source_whisper_upper_bound": "must_be_recomputed_by_evaluator",
        })
    report = {
        "status": "evaluation_input_only",
        "rows": manifest,
        "license": "CC BY-NC-SA 4.0",
        "source_audio_committed": False,
        "release_allowed": False,
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
