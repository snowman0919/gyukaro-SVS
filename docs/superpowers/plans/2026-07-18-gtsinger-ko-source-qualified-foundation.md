# GTSinger KO Source-Qualified Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the fail-closed source qualification and Korean DiffSinger foundation gate required before multilingual or GYU identity training.

**Architecture:** One preparation script reuses the repository's pinned GTSinger, Whisper, MMS-CTC, audio-analysis, and DiffSinger paths to qualify KO-Soprano-2 rows and generate training inputs. One evaluator reuses existing score, RMVPE, artifact, and transcript helpers to render and gate the three fixed checkpoints across seven Korean stress cases and seeds 7/21/42. If either source or foundation gate fails, write a rejection report and stop before identity work.

**Tech Stack:** Python 3.11, pytest, PyTorch, torchaudio MMS_FA, Transformers Whisper large-v3-turbo, soundfile, OpenVPI DiffSinger revision `753b7cc622aadf802b3145d7bb8f7df4afa213c4`, RMVPE, JSON/JSONL/YAML.

## Global Constraints

- Authoritative design: `docs/superpowers/specs/2026-07-18-source-qualified-multilingual-gyu-svs-design.md`.
- GTSinger revision: `4426c862beed558b7e1cb8a4dce7e8c0c83bb208`.
- Local non-commercial experiment only; public release is not authorized.
- Start with KO-Soprano-2 Control_Group; do not download or train KO-Tenor-1.
- Keep `data/source/` unchanged and never commit source audio, external datasets, checkpoints, rendered WAVs, caches, or large plots.
- Whisper is required but is not the sole alignment authority.
- Training is forbidden unless every frozen source-corpus minimum passes.
- Identity, multilingual, production renderer, package, and OpenUtau code remain unchanged in this plan.
- Maximum acoustic training is 15,000 steps; inspect only steps 5,000, 10,000, and 15,000.
- Every Korean stress case is rendered with seeds 7, 21, and 42.
- One mandatory row failure rejects the complete foundation.

---

### Task 1: Freeze the source qualification contract

**Files:**
- Create: `configs/gtsinger_ko_qualified_protocol.json`
- Create: `scripts/prepare_diffsinger_gtsinger_ko_qualified.py`
- Create: `tests/test_diffsinger_gtsinger_ko_qualified.py`

**Interfaces:**
- Produces: `normalized_text(text: str) -> str`
- Produces: `normalized_phones(row: dict) -> list[str]`
- Produces: `row_rejections(row: dict, measured: dict, gates: dict) -> list[str]`
- Produces: `corpus_summary(rows: list[dict], minimums: dict) -> dict`
- Produces: `song_splits(rows: list[dict]) -> dict[str, list[str]]`

- [ ] **Step 1: Add the frozen protocol**

Create `configs/gtsinger_ko_qualified_protocol.json` with this exact content:

```json
{
  "schema": 1,
  "dataset": "GTSinger/GTSinger",
  "dataset_revision": "4426c862beed558b7e1cb8a4dce7e8c0c83bb208",
  "dataset_license": "CC-BY-NC-SA-4.0",
  "language": "Korean",
  "singer": "KO-Soprano-2",
  "group": "Control_Group",
  "diffsinger_revision": "753b7cc622aadf802b3145d7bb8f7df4afa213c4",
  "seeds": [7, 21, 42],
  "checkpoints": [5000, 10000, 15000],
  "row_gates": {
    "duration_min_seconds": 2.0,
    "duration_max_seconds": 15.0,
    "duration_delta_max_seconds": 0.05,
    "clipping_samples_max": 0,
    "whisper_similarity_min": 0.8,
    "ctc_coverage_min": 0.9,
    "ctc_unknown_ratio_max": 0.05,
    "ctc_monotonic_required": true
  },
  "corpus_minimums": {
    "rows": 200,
    "duration_seconds": 1800.0,
    "fast_rows": 40,
    "high_register_rows": 20,
    "sustained_rows": 20,
    "large_interval_rows": 20
  },
  "foundation_gates": {
    "lyric_similarity_min": 0.9,
    "pitch_mae_mean_max_cents": 20.0,
    "pitch_mae_row_max_cents": 35.0,
    "pitch_p90_max_cents": 60.0,
    "gross_pitch_error_max": 0.03,
    "voicing_accuracy_min": 0.9,
    "clipping_samples_max": 0,
    "hf_spike_ratio_to_source_max": 1.1,
    "sample_jump_ratio_to_source_max": 1.1,
    "waveform_discontinuity_ratio_to_source_max": 1.1,
    "stft_spike_ratio_to_source_max": 1.1
  }
}
```

- [ ] **Step 2: Write failing tests for row acceptance and rejection**

Add imports and fixtures to `tests/test_diffsinger_gtsinger_ko_qualified.py`:

```python
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_diffsinger_gtsinger_ko_qualified import (
    corpus_summary,
    normalized_phones,
    normalized_text,
    row_rejections,
    song_splits,
)

GATES = json.loads((ROOT / "configs/gtsinger_ko_qualified_protocol.json").read_text())["row_gates"]


def source_row(name="Korean#KO-Soprano-2#Breathy#song-a#Control_Group#0000"):
    return {
        "item_name": name,
        "language": "Korean",
        "singer": "KO-Soprano-2",
        "pace": "fast",
        "range": "high",
        "txt": ["가", "나"],
        "ph": ["k_ko", "ɐ_ko", "n_ko", "ɐ_ko"],
        "ph_durs": [0.1, 0.9, 0.1, 0.9],
        "ep_pitches": [60, 60, 72, 72],
        "ep_notedurs": [1.0, 1.0, 1.0, 1.0],
        "wav_fn": "Korean/test.wav"
    }


def measured(**updates):
    value = {
        "audio_duration_seconds": 2.0,
        "clipping_samples": 0,
        "whisper_similarity": 0.9,
        "ctc_coverage": 0.95,
        "ctc_unknown_ratio": 0.0,
        "ctc_monotonic": True,
    }
    value.update(updates)
    return value


def test_source_row_passes_only_complete_multi_evidence_gate():
    assert row_rejections(source_row(), measured(), GATES) == []


@pytest.mark.parametrize(("updates", "reason"), [
    ({"audio_duration_seconds": 1.9}, "duration"),
    ({"clipping_samples": 1}, "clipping"),
    ({"whisper_similarity": 0.79}, "whisper"),
    ({"ctc_coverage": 0.89}, "ctc_coverage"),
    ({"ctc_unknown_ratio": 0.051}, "ctc_unknown"),
    ({"ctc_monotonic": False}, "ctc_monotonic"),
])
def test_source_row_rejects_each_failed_gate(updates, reason):
    assert reason in row_rejections(source_row(), measured(**updates), GATES)


def test_metadata_shape_and_duration_mismatch_are_rejected():
    row = source_row()
    row["ep_pitches"] = row["ep_pitches"][:-1]
    assert "metadata_shape" in row_rejections(row, measured(), GATES)
    assert "duration_alignment" in row_rejections(
        source_row(), measured(audio_duration_seconds=2.051), GATES
    )


def test_text_and_phone_normalization_are_stable():
    assert normalized_text(" 가, 나! ") == "가나"
    row = source_row()
    row["ph"] = ["<SP>", "k_ko", "ɐ_ko", "<AP>"]
    assert normalized_phones(row) == ["SP", "k_ko", "ɐ_ko", "AP"]
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_diffsinger_gtsinger_ko_qualified.py
```

Expected: collection fails because `prepare_diffsinger_gtsinger_ko_qualified` does not exist.

- [ ] **Step 4: Implement the minimum pure qualification functions**

Create `scripts/prepare_diffsinger_gtsinger_ko_qualified.py` with these definitions before adding runtime code:

```python
#!/usr/bin/env python3
"""Qualify GTSinger KO source rows before any DiffSinger training."""
from __future__ import annotations

import re


def normalized_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def normalized_phones(row: dict) -> list[str]:
    return ["SP" if value == "<SP>" else "AP" if value == "<AP>" else value
            for value in row["ph"]]


def row_rejections(row: dict, measured: dict, gates: dict) -> list[str]:
    reasons = []
    arrays = (row.get("ph"), row.get("ph_durs"), row.get("ep_pitches"), row.get("ep_notedurs"))
    if not all(isinstance(value, (list, tuple)) and value for value in arrays) or len({len(value) for value in arrays}) != 1:
        reasons.append("metadata_shape")
    duration = float(measured["audio_duration_seconds"])
    if not gates["duration_min_seconds"] <= duration <= gates["duration_max_seconds"]:
        reasons.append("duration")
    if row.get("ph_durs") and abs(sum(map(float, row["ph_durs"])) - duration) > gates["duration_delta_max_seconds"]:
        reasons.append("duration_alignment")
    if measured["clipping_samples"] > gates["clipping_samples_max"]:
        reasons.append("clipping")
    if measured["whisper_similarity"] < gates["whisper_similarity_min"]:
        reasons.append("whisper")
    if measured["ctc_coverage"] < gates["ctc_coverage_min"]:
        reasons.append("ctc_coverage")
    if measured["ctc_unknown_ratio"] > gates["ctc_unknown_ratio_max"]:
        reasons.append("ctc_unknown")
    if gates["ctc_monotonic_required"] and not measured["ctc_monotonic"]:
        reasons.append("ctc_monotonic")
    if not any(float(value) > 0 for value in row.get("ep_pitches", [])):
        reasons.append("no_voiced_note")
    if not any(value not in ("<SP>", "<AP>") for value in row.get("ph", [])):
        reasons.append("no_lexical_phone")
    return reasons


def corpus_summary(rows: list[dict], minimums: dict) -> dict:
    counts = {
        "rows": len(rows),
        "duration_seconds": sum(float(row["audio_duration_seconds"]) for row in rows),
        "fast_rows": sum(row["pace"] == "fast" for row in rows),
        "high_register_rows": sum(row["range"] == "high" for row in rows),
        "sustained_rows": sum(float(row["max_phone_duration_seconds"]) >= 1.0 for row in rows),
        "large_interval_rows": sum(float(row["max_interval_semitones"]) >= 7.0 for row in rows),
    }
    failures = [name for name, threshold in minimums.items() if counts[name] < threshold]
    return {"status": "source_qualification_pass" if not failures else "foundation_source_gate_reject",
            "counts": counts, "failed_minimums": failures, "training_allowed": not failures}


def song_splits(rows: list[dict]) -> dict[str, list[str]]:
    songs = sorted({row["item_name"].split("#")[3] for row in rows})
    heldout = set(songs[-2:])
    validation = set(songs[-4:-2])
    return {
        split: sorted(row["item_name"] for row in rows if (
            row["item_name"].split("#")[3] in heldout if split == "test" else
            row["item_name"].split("#")[3] in validation if split == "validation" else
            row["item_name"].split("#")[3] not in heldout | validation
        ))
        for split in ("train", "validation", "test")
    }
```

- [ ] **Step 5: Add and pass corpus/split tests**

Append:

```python
def test_corpus_minimum_is_fail_closed():
    row = source_row() | {
        "audio_duration_seconds": 10.0,
        "max_phone_duration_seconds": 1.2,
        "max_interval_semitones": 12.0,
    }
    minimums = {"rows": 2, "duration_seconds": 20.0, "fast_rows": 2,
                "high_register_rows": 2, "sustained_rows": 2, "large_interval_rows": 2}
    assert corpus_summary([row], minimums)["status"] == "foundation_source_gate_reject"
    assert corpus_summary([row, row | {"item_name": row["item_name"] + "x"}], minimums)["training_allowed"] is True


def test_song_split_never_leaks_complete_song():
    rows = [source_row(f"Korean#KO-Soprano-2#Breathy#song-{index}#Control_Group#0000")
            for index in range(6)]
    splits = song_splits(rows)
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    by_song = {name.split("#")[3]: split for split, names in splits.items() for name in names}
    assert len(by_song) == 6
```

Run:

```bash
pytest -q tests/test_diffsinger_gtsinger_ko_qualified.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the frozen pure gate**

```bash
git add configs/gtsinger_ko_qualified_protocol.json scripts/prepare_diffsinger_gtsinger_ko_qualified.py tests/test_diffsinger_gtsinger_ko_qualified.py
git commit -m "test(svs): freeze Korean source qualification"
```

---

### Task 2: Execute Whisper, CTC, and waveform source qualification

**Files:**
- Modify: `scripts/prepare_diffsinger_gtsinger_ko_qualified.py`
- Modify: `tests/test_diffsinger_gtsinger_ko_qualified.py`
- Generate and commit on execution: `data/manifests/gtsinger_ko_source_qualified.jsonl`
- Generate and commit on execution: `artifacts/reports/gtsinger_ko_source_qualification/summary.json`
- Generate locally only: `data/external/work/gtsinger_ko_source_qualification/all_rows.jsonl`

**Interfaces:**
- Produces: `ctc_metrics(log_probs, target_ids, unknown_count=0) -> dict`
- Produces CLI: `python scripts/prepare_diffsinger_gtsinger_ko_qualified.py --download --qualify`

- [ ] **Step 1: Write a failing test for CTC evidence shape**

Append:

```python
import torch
from prepare_diffsinger_gtsinger_ko_qualified import ctc_metrics


def test_ctc_metrics_require_complete_monotonic_target():
    log_probs = torch.log_softmax(torch.tensor([[[8., 0., 0.], [0., 8., 0.],
                                                 [8., 0., 0.], [0., 0., 8.]]]), -1)
    result = ctc_metrics(log_probs, torch.tensor([[1, 2]]))
    assert result["ctc_monotonic"] is True
    assert result["ctc_unknown_ratio"] == 0.0
    assert 0.0 < result["ctc_coverage"] <= 1.0
```

Run the focused test and confirm it fails because `ctc_metrics` is absent.

- [ ] **Step 2: Implement CTC evidence and the bounded runtime**

Add imports for `argparse`, `hashlib`, `json`, `math`, `os`, `subprocess`, `Path`, `SequenceMatcher`, `numpy`, `soundfile`, `torch`, `torchaudio`, `resample_poly`, `hf_hub_download`, and the existing `mms_alignment_target` from `gyu_singer.experiments.korean_phones`.

Implement `ctc_metrics` exactly around torchaudio's forced alignment:

```python
def ctc_metrics(log_probs: torch.Tensor, target_ids: torch.Tensor, unknown_count: int = 0) -> dict:
    alignment, scores = torchaudio.functional.forced_align(log_probs, target_ids)
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    frames = log_probs.shape[1]
    monotonic = len(spans) == target_ids.shape[1] and all(
        left.end <= right.start for left, right in zip(spans, spans[1:])
    )
    target_count = target_ids.shape[1] + unknown_count
    return {
        "ctc_coverage": float(len(spans) / target_count),
        "ctc_unknown_ratio": float(unknown_count / target_count),
        "ctc_monotonic": monotonic,
        "ctc_aligned_frame_ratio": float(sum(span.end - span.start for span in spans) / frames),
        "ctc_mean_log_score": float(sum(float(span.score) for span in spans) / len(spans)),
    }
```

Add a `--download --qualify` CLI that:

1. sparse-checks out only `Korean/KO-Soprano-2/**/Control_Group/*.wav`, `processed/Korean/metadata.json`, and `dataset_license.md` at the pinned revision using the same Git-LFS pattern as `prepare_diffsinger_gtsinger_ja.py`;
2. filters metadata to Korean, KO-Soprano-2, and Control_Group before loading models;
3. loads Whisper once and MMS_FA once;
4. measures audio duration, clipping samples, SHA-256, free Korean Whisper transcript/similarity, target-conditioned CTC coverage/unknown ratio/monotonicity, maximum phone duration, and maximum adjacent voiced-note interval;
5. applies `row_rejections` without changing thresholds;
6. writes all rows under `data/external/work/` and accepted rows to the compact manifest;
7. computes `corpus_summary` and writes the compact summary;
8. exits with code 2 when the source gate rejects, and 0 when it passes.

Use `SequenceMatcher(None, normalized_text(expected), normalized_text(transcript)).ratio()` and pass Whisper `attention_mask`, `language="ko"`, `task="transcribe"`. The accepted manifest must retain `label_status: "dataset_metadata_plus_measured_whisper_and_target_conditioned_mms_ctc"` and must never call its alignment human-verified.

The summary also freezes global p95 accepted-source baselines for HF spike, sample jump, waveform discontinuity, and spectral spike at FFT 256/1024/4096. Task 4 divides candidate values by these frozen p95 values; a missing or zero baseline is a gate error.

- [ ] **Step 3: Verify the focused tests pass**

```bash
pytest -q tests/test_diffsinger_gtsinger_ko_qualified.py
```

Expected: all tests pass, including the synthetic CTC path.

- [ ] **Step 4: Commit the qualification runner before running external data**

```bash
git diff --check
git add scripts/prepare_diffsinger_gtsinger_ko_qualified.py tests/test_diffsinger_gtsinger_ko_qualified.py
git commit -m "feat(svs): qualify GTSinger Korean sources"
```

---

### Task 3: Generate DiffSinger training data without split leakage

**Files:**
- Modify: `scripts/prepare_diffsinger_gtsinger_ko_qualified.py`
- Modify: `tests/test_diffsinger_gtsinger_ko_qualified.py`
- Generate and commit after a source pass: `configs/diffsinger_gtsinger_ko_qualified.yaml`
- Generate locally only: `data/external/work/diffsinger_score_native/raw/gtsinger_ko_qualified/`

**Interfaces:**
- Produces: `write_training_data(rows: list[dict], dataset_root: Path, output: Path) -> dict`
- Produces: `build_config(summary: dict, raw_dir: Path, dictionary: Path) -> dict`
- Consumes: accepted source manifest from Task 2.

- [ ] **Step 1: Write failing tests for deterministic data export**

Append tests that create two temporary WAVs with `soundfile.write`, call `write_training_data`, and assert:

```python
import csv
import numpy as np
import soundfile as sf
from prepare_diffsinger_gtsinger_ko_qualified import write_training_data


def test_training_export_preserves_phone_duration_and_song_split(tmp_path):
    rows = []
    for index in range(6):
        row = source_row(f"Korean#KO-Soprano-2#Breathy#song-{index}#Control_Group#0000")
        source = tmp_path / f"source-{index}.wav"
        sf.write(source, np.zeros(48000 * 2, dtype=np.float32), 48000)
        rows.append(row | {"source_path": str(source), "audio_duration_seconds": 2.0})
    result = write_training_data(rows, tmp_path, tmp_path / "raw")
    csv_rows = list(csv.DictReader((tmp_path / "raw/transcriptions.csv").open()))
    assert len(csv_rows) == 6
    assert all(len(row["ph_seq"].split()) == len(row["ph_dur"].split()) for row in csv_rows)
    assert result["song_split_leakage"] is False
    assert result["rows"] == 6
```

Run the focused test and verify it fails because `write_training_data` is absent.

- [ ] **Step 2: Implement export by reusing the Japanese preparation pattern**

Implement `write_training_data` with the same CSV fields, symlink behavior, phone normalization, and sub-2-ms duration repair used by `prepare_diffsinger_gtsinger_ja.py`. Derive stable names from a SHA-256 sort of `item_name`, write `split.json`, and fail if one complete song appears in more than one split.

Implement `build_config` by loading `configs/diffsinger_pjs_compact.yaml` and overriding only:

```python
{
    "dictionaries": {"ko": str(dictionary)},
    "datasets": [{
        "raw_data_dir": str(raw_dir),
        "speaker": "gts_ko_soprano_2",
        "spk_id": 0,
        "language": "ko",
        "test_prefixes": validation_and_test_names,
    }],
    "binary_data_dir": str(work / "binary_gtsinger_ko_qualified"),
    "binarizer_cls": "diffsinger_neutral_augmentation_binarizer.NeutralAugmentationBinarizer",
    "finetune_enabled": False,
    "finetune_ckpt_path": None,
    "max_updates": 15000,
    "val_check_interval": 1000,
    "num_ckpt_keep": 5,
    "work_dir": str(cache / "diffsinger/checkpoints/gtsinger_ko_qualified"),
}
```

Retain the Japanese foundation's 256-dimension encoder, 384-channel auxiliary decoder, score/F0 path, augmentation settings, and pinned vocoder. Do not add a new model component.

- [ ] **Step 3: Pass focused and existing preparation tests**

```bash
pytest -q tests/test_diffsinger_gtsinger_ko_qualified.py tests/test_diffsinger_pjs.py
```

Expected: all tests pass.

- [ ] **Step 4: Commit training-data generation**

```bash
git diff --check
git add scripts/prepare_diffsinger_gtsinger_ko_qualified.py tests/test_diffsinger_gtsinger_ko_qualified.py
git commit -m "feat(svs): prepare qualified Korean foundation"
```

---

### Task 4: Add the strict Korean foundation matrix evaluator

**Files:**
- Create: `examples/review_repeated_ko.json`
- Create: `examples/review_high_ko.json`
- Create: `scripts/evaluate_diffsinger_gtsinger_ko_foundation.py`
- Create: `tests/test_diffsinger_gtsinger_ko_foundation.py`

**Interfaces:**
- Produces: `gate_row(row: dict, gates: dict) -> list[str]`
- Produces: `summarize_checkpoint(rows: list[dict], protocol: dict) -> dict`
- Produces: `select_checkpoint(reports: list[dict]) -> dict`
- Produces CLI stages: `freeze`, `render`, `evaluate`, and `report`
- Reuses: `build_score_ds`, `transcript_flags`, and `distribution` from `run_gtsinger_gyu_identity_diagnostic.py`
- Reuses: `pitch_errors` and `voicing_accuracy` from `evaluate_diffsinger_pjs_rapid.py`
- Reuses: `audio16` and `acoustics` from `evaluate_rc4_artifact_matrix.py`

- [ ] **Step 1: Add fixed repeated-note and high-register scores**

Use project-authored Korean lyrics and one phrase-level score per file:

```json
{"language":"ko","tempo":150,"sample_rate":48000,"notes":[{"pitch":64,"start":0.0,"duration":0.4,"lyric":"같"},{"pitch":64,"start":0.4,"duration":0.4,"lyric":"은"},{"pitch":64,"start":0.8,"duration":0.4,"lyric":"음"},{"pitch":64,"start":1.2,"duration":0.4,"lyric":"을"},{"pitch":64,"start":1.6,"duration":0.4,"lyric":"다"},{"pitch":64,"start":2.0,"duration":0.4,"lyric":"시"},{"pitch":64,"start":2.4,"duration":0.6,"lyric":"불러"}]}
```

```json
{"language":"ko","tempo":110,"sample_rate":48000,"notes":[{"pitch":72,"start":0.0,"duration":0.8,"lyric":"높"},{"pitch":76,"start":0.8,"duration":0.8,"lyric":"은"},{"pitch":79,"start":1.6,"duration":1.0,"lyric":"하늘"},{"pitch":76,"start":2.6,"duration":0.8,"lyric":"위"},{"pitch":72,"start":3.4,"duration":1.0,"lyric":"로"}]}
```

- [ ] **Step 2: Write failing strict-gate tests**

Create synthetic rows for seven cases and three seeds. Test that:

- all 21 rows pass only when every threshold passes;
- one lyric similarity of `0.899` rejects the checkpoint;
- one pitch MAE of `35.01` rejects it;
- aggregate mean pitch MAE of `20.01` rejects it;
- one voicing accuracy of `0.899` rejects it;
- clipping, repetition, omission, HF, sample-jump, or STFT regression rejects it;
- a missing case/seed or non-finite metric raises `ValueError`;
- selection first filters passing reports, then chooses highest minimum lyric similarity, then lowest maximum pitch error, then earliest step.

The core assertion is:

```python
def test_one_failed_seed_rejects_complete_checkpoint():
    rows = passing_matrix()
    rows[-1]["lyric_similarity"] = 0.899
    result = summarize_checkpoint(rows, protocol())
    assert result["status"] == "foundation_ko_gate_reject"
    assert result["pass_count"] == 20
    assert result["training_identity_allowed"] is False
```

Run `pytest -q tests/test_diffsinger_gtsinger_ko_foundation.py` and verify RED because the evaluator is absent.

- [ ] **Step 3: Implement the strict evaluator and renderer orchestration**

Implement the pure gate first. Require these per-row fields:

```python
REQUIRED = (
    "lyric_similarity", "pitch_mae_cents", "pitch_p90_abs_cents",
    "gross_pitch_error_rate", "voicing_accuracy", "clipping_samples",
    "hf_spike_ratio_to_source", "sample_jump_ratio_to_source",
    "waveform_discontinuity_ratio_to_source", "stft_spike_ratio_to_source",
)
```

`summarize_checkpoint` must validate exactly seven case IDs times three seeds, reject non-finite values, apply every per-row gate, additionally enforce mean pitch MAE <= 20 cents, and return complete metric distributions.

Implement the pure decision core as follows; import `math`, `numpy as np`, and `distribution` from `run_gtsinger_gyu_identity_diagnostic`:

```python
CASES = ("quality_ko", "rapid_ko", "large_interval_ko", "sustain_ko",
         "repeated_ko", "high_ko", "phrase_boundary_ko")


def gate_row(row: dict, gates: dict) -> list[str]:
    failures = []
    if (row["lyric_similarity"] < gates["lyric_similarity_min"]
            or row.get("repetition_detected") or row.get("omission_detected")):
        failures.append("foundation_content_failure")
    if (row["pitch_mae_cents"] > gates["pitch_mae_row_max_cents"]
            or row["pitch_p90_abs_cents"] > gates["pitch_p90_max_cents"]
            or row["gross_pitch_error_rate"] > gates["gross_pitch_error_max"]):
        failures.append("pitch_regression")
    if row["voicing_accuracy"] < gates["voicing_accuracy_min"]:
        failures.append("voicing_regression")
    if row["clipping_samples"] > gates["clipping_samples_max"]:
        failures.append("clipping_failure")
    artifact_checks = {
        "hf_spike_ratio_to_source": "hf_spike_ratio_to_source_max",
        "sample_jump_ratio_to_source": "sample_jump_ratio_to_source_max",
        "waveform_discontinuity_ratio_to_source": "waveform_discontinuity_ratio_to_source_max",
        "stft_spike_ratio_to_source": "stft_spike_ratio_to_source_max",
    }
    if any(row[name] > gates[limit] for name, limit in artifact_checks.items()):
        failures.append("artifact_regression")
    return failures


def summarize_checkpoint(rows: list[dict], protocol: dict) -> dict:
    expected = {(case, seed) for case in CASES for seed in protocol["seeds"]}
    actual = {(row.get("case"), row.get("seed")) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("foundation matrix mismatch")
    for row in rows:
        for name in REQUIRED:
            if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]):
                raise ValueError(f"missing or non-finite metric: {name}")
        if not row.get("whisper_transcript") or len(row.get("audio_sha256", "")) != 64:
            raise ValueError("missing transcript or WAV SHA")
    gates = protocol["foundation_gates"]
    decisions = [{"case": row["case"], "seed": row["seed"],
                  "failures": gate_row(row, gates)} for row in rows]
    mean_pitch = float(np.mean([row["pitch_mae_cents"] for row in rows]))
    aggregate_failure = mean_pitch > gates["pitch_mae_mean_max_cents"]
    pass_count = sum(not item["failures"] for item in decisions)
    passed = pass_count == len(rows) and not aggregate_failure
    return {
        "step": rows[0]["checkpoint_step"],
        "status": "foundation_ko_gate_pass" if passed else "foundation_ko_gate_reject",
        "pass_count": pass_count,
        "pass_ratio": pass_count / len(rows),
        "training_identity_allowed": passed,
        "aggregate_pitch_failure": aggregate_failure,
        "decisions": decisions,
        "metrics": {name: distribution([float(row[name]) for row in rows]) for name in REQUIRED},
        "minimum_lyric_similarity": min(row["lyric_similarity"] for row in rows),
        "maximum_pitch_mae_cents": max(row["pitch_mae_cents"] for row in rows),
    }


def select_checkpoint(reports: list[dict]) -> dict:
    passed = [report for report in reports if report["status"] == "foundation_ko_gate_pass"]
    if not passed:
        return {"status": "foundation_ko_gate_reject", "selected_step": None}
    selected = max(passed, key=lambda report: (
        report["minimum_lyric_similarity"],
        -report["maximum_pitch_mae_cents"],
        -report["step"],
    ))
    return {"status": "foundation_ko_gate_pass", "selected_step": selected["step"]}
```

The CLI must:

1. `freeze`: hash the protocol, config, dictionary, three checkpoints, seven score files, and source-qualification manifest; write seven `.ds` inputs with `build_score_ds`;
2. `render`: invoke the existing `data/cache/diffsinger/scripts/infer.py acoustic` path for every checkpoint/case/seed using speaker `gts_ko_soprano_2`, `--depth 0`, and no post-processing;
3. `evaluate`: run free Korean Whisper, RMVPE, voicing, clipping, HF, sample jump, waveform discontinuity, and FFT 256/1024/4096 spectral-spike measurements for every actual WAV; divide artifact values by the frozen accepted-source p95 values and record path and SHA-256;
4. `report`: select or reject using validation evidence only and write `artifacts/reports/gtsinger_ko_qualified_foundation/evaluation.json`.

Every failed row gets a local waveform and three-resolution STFT plot under the ignored report work directory. No plot is needed for every passing row.

- [ ] **Step 4: Pass focused and legacy gate tests**

```bash
pytest -q tests/test_diffsinger_gtsinger_ko_foundation.py tests/test_diffsinger_heldout_gate.py tests/test_gtsinger_gyu_identity_diagnostic.py
```

Expected: all tests pass without changing previous rejection decisions.

- [ ] **Step 5: Commit the evaluator**

```bash
git diff --check
git add examples/review_repeated_ko.json examples/review_high_ko.json scripts/evaluate_diffsinger_gtsinger_ko_foundation.py tests/test_diffsinger_gtsinger_ko_foundation.py
git commit -m "test(svs): gate Korean score-native foundation"
```

---

### Task 5: Run the frozen source qualification gate

**Files:**
- Generate: `data/manifests/gtsinger_ko_source_qualified.jsonl`
- Generate: `artifacts/reports/gtsinger_ko_source_qualification/summary.json`
- Generate only on pass: `configs/diffsinger_gtsinger_ko_qualified.yaml`
- Generate locally: `data/external/raw/gtsinger-lfs/`
- Generate locally: `data/external/work/gtsinger_ko_source_qualification/`

- [ ] **Step 1: Record environment and free-space evidence**

Run and save the compact values in the summary:

```bash
python -V
python -c "import torch, torchaudio, transformers; print(torch.__version__, torch.version.cuda, torchaudio.__version__, transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h . data/cache data/external
git -C data/cache/diffsinger rev-parse HEAD
```

- [ ] **Step 2: Download and qualify only the approved subset**

```bash
python scripts/prepare_diffsinger_gtsinger_ko_qualified.py --download --qualify
```

Expected: either `source_qualification_pass` with every minimum satisfied, or exit code 2 with `foundation_source_gate_reject` and named failed minimums.

- [ ] **Step 3: Apply the mandatory early stop**

If rejected, do not run binarization or training. Write `docs/gtsinger_ko_qualified_foundation.md` with first line `NOT A RELEASE REPORT — FOUNDATION SOURCE REJECTED`, include counts, failed minimums, actual local evidence paths, and the next valid requirement: rights-controlled GYU score-native recording.

If passed, run the same script with `--prepare`, verify the generated config contains `max_updates: 15000`, and continue to Task 6.

- [ ] **Step 4: Commit compact source evidence**

```bash
git diff --check
git add data/manifests/gtsinger_ko_source_qualified.jsonl artifacts/reports/gtsinger_ko_source_qualification/summary.json
git add configs/diffsinger_gtsinger_ko_qualified.yaml 2>/dev/null || true
git commit -m "data(svs): freeze qualified Korean sources"
```

---

### Task 6: Conditionally train and evaluate the Korean foundation

**Files:**
- Generate locally: `data/external/work/diffsinger_score_native/binary_gtsinger_ko_qualified/`
- Generate locally: `data/cache/diffsinger/checkpoints/gtsinger_ko_qualified/`
- Generate locally: `data/external/work/gtsinger_ko_qualified_foundation/`
- Generate and commit: `artifacts/reports/gtsinger_ko_qualified_foundation/evaluation.json`

- [ ] **Step 1: Verify the source gate before expensive work**

```bash
python -c "import json; p=json.load(open('artifacts/reports/gtsinger_ko_source_qualification/summary.json')); assert p['status']=='source_qualification_pass' and p['training_allowed']"
```

Expected: exit 0. A failed assertion forbids this task.

- [ ] **Step 2: Binarize with the pinned DiffSinger environment**

```bash
cd data/cache/diffsinger
.venv/bin/python scripts/binarize.py --config /home/kotori9/code/gyukaro/.worktrees/svs-05-voicebank-factory/configs/diffsinger_gtsinger_ko_qualified.yaml
```

Expected: train/valid binary data is created without duration or phone errors.

- [ ] **Step 3: Train once to the frozen 15,000-step ceiling**

```bash
cd data/cache/diffsinger
.venv/bin/python scripts/train.py --config /home/kotori9/code/gyukaro/.worktrees/svs-05-voicebank-factory/configs/diffsinger_gtsinger_ko_qualified.yaml --reset
```

Expected: checkpoints at steps 5,000, 10,000, and 15,000 exist. Do not extend the run or alter a loss weight after viewing results.

- [ ] **Step 4: Freeze, render, and evaluate all 63 outputs**

```bash
python scripts/evaluate_diffsinger_gtsinger_ko_foundation.py freeze
python scripts/evaluate_diffsinger_gtsinger_ko_foundation.py render
python scripts/evaluate_diffsinger_gtsinger_ko_foundation.py evaluate
python scripts/evaluate_diffsinger_gtsinger_ko_foundation.py report
```

Expected: `63` actual WAV paths and SHA-256 values, complete transcripts and metrics, and exactly one of `foundation_ko_gate_pass` or `foundation_ko_gate_reject`.

- [ ] **Step 5: Enforce the outcome**

If rejected, preserve checkpoints and WAVs only as ignored diagnostic evidence and do not create multilingual or identity plans from the rejected checkpoint.

If passed, record the selected checkpoint hash and label it `qualified_korean_score_native_foundation_only`. Do not call it GYU, production-ready, package-ready, or human-approved.

---

### Task 7: Report and verify the bounded phase

**Files:**
- Create: `docs/gtsinger_ko_qualified_foundation.md`
- Modify only when evidence changes: `configs/research_evidence.json`
- Modify only when evidence changes: `configs/project_status.json`

- [ ] **Step 1: Write the truthful report from generated JSON**

The first line must be one of:

```text
NOT A RELEASE REPORT — FOUNDATION SOURCE REJECTED
```

```text
NOT A RELEASE REPORT — KOREAN FOUNDATION REJECTED
```

```text
NOT A RELEASE REPORT — KOREAN FOUNDATION QUALIFIED ONLY
```

Include source counts, coverage, split hashes, environment, training history, checkpoint selection, 7x3 per-checkpoint metrics, failure taxonomy, local WAV paths/SHA-256, plots, and explicit multilingual/identity/runtime/package status.

- [ ] **Step 2: Run focused, full, and dataset verification**

```bash
pytest -q tests/test_diffsinger_gtsinger_ko_qualified.py tests/test_diffsinger_gtsinger_ko_foundation.py
pytest -q
python scripts/validate_dataset.py
git diff --check
git status --short
```

Expected: all tests pass; dataset reports `132 recordings`, `corrupt 0`; diff check is clean; no WAV, checkpoint, cache, or external dataset is staged.

- [ ] **Step 3: Verify protected runtime/package state**

```bash
git diff 9b443ee -- src/gyu_singer configs/release_gate.json scripts/package_openutau_diffsinger_candidate.py
git status --short | rg '\.(wav|ckpt|pt|pth|onnx)$' && exit 1 || true
```

Expected: no production renderer or package pointer change and no model/audio artifact staged.

- [ ] **Step 4: Commit the report and compact evidence**

```bash
git add docs/gtsinger_ko_qualified_foundation.md artifacts/reports/gtsinger_ko_source_qualification/summary.json
git add artifacts/reports/gtsinger_ko_qualified_foundation/evaluation.json 2>/dev/null || true
git add configs/research_evidence.json configs/project_status.json
git commit -m "docs(svs): report Korean foundation gate"
```

- [ ] **Step 5: Decide the next plan from the gate**

On `foundation_ko_gate_pass`, write the separate multilingual-foundation and bounded-GYU-identity implementation plan from the approved design. On rejection, keep the full user goal incomplete and report the exact data or foundation failure without manufacturing a package.
