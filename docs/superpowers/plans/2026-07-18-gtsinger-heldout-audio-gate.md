# GTSinger Held-Out Audio Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a five-phrase, source-qualified DiffSinger gate that always records free Whisper, RMVPE, waveform, and multi-reference GYU identity evidence before a candidate is reported.

**Architecture:** A small builder converts pinned GTSinger manual alignments and RMVPE tracks into reproducible `.ds` rows and a manifest. The existing evaluator gains repeated identity references so each rendered phrase is analyzed once against five GYU recordings; no runtime or model code changes.

**Tech Stack:** Python 3.11, NumPy, soundfile, PyTorch, RMVPE, Whisper large-v3-turbo, WavLM, ECAPA, pytest, existing DiffSinger CLI.

## Global Constraints

- Keep `data/source/` unchanged and never commit source recordings.
- Mark GTSinger alignment as dataset-provided and F0 as RMVPE-derived; never call it GYU supervision.
- Preserve RC7 byte-for-byte and do not change the production runtime or package.
- Every reported WAV requires SHA-256, free Whisper, RMVPE F0/voicing, clipping, waveform, and spectral evidence.
- Human listening remains mandatory; an automated pass is not RC8/RC9 acceptance.
- GTSinger-derived checkpoints remain CC BY-NC-SA 4.0 and are not release-authorized.

---

### Task 1: Reproducible five-phrase `.ds` builder

**Files:**
- Create: `scripts/build_diffsinger_gtsinger_heldout_set.py`
- Create: `tests/test_diffsinger_heldout_gate.py`
- Generate locally: `data/external/work/gtsinger/heldout_eval/*.ds`
- Generate locally: `artifacts/reports/diffsinger_gtsinger_heldout_set/manifest.json`

**Interfaces:**
- Consumes: pinned GTSinger metadata, source WAVs, `selected_rows()` and `normalized_phones()` from `scripts/prepare_diffsinger_gtsinger_ja.py`, local RMVPE.
- Produces: `build_ds_row(row: dict, audio_duration: float, f0: np.ndarray) -> dict` and a manifest for IDs `gtsja0165`, `gtsja0172`, `gtsja0174`, `gtsja0379`, `gtsja0380`.

- [ ] **Step 1: Write the failing pure-function test**

```python
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
from build_diffsinger_gtsinger_heldout_set import build_ds_row

def test_heldout_ds_row_preserves_manual_timing_and_rmvpe_grid():
    row = {"ph": ["i_ja", "<AP>"], "ph_durs": [.08, .02], "txt": ["い", "<AP>"]}
    result = build_ds_row(row, .10, np.array([261.63] * 5, dtype=np.float32))
    assert result["ph_seq"] == "i_ja AP"
    assert result["ph_dur"] == "0.0800000 0.0200000"
    assert result["text"] == "い"
    assert result["f0_timestep"] == 0.02
    assert len(result["f0_seq"].split()) == 5
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run: `.venv-diffsinger/bin/pytest tests/test_diffsinger_heldout_gate.py::test_heldout_ds_row_preserves_manual_timing_and_rmvpe_grid -q`

Expected: `ModuleNotFoundError: No module named 'build_diffsinger_gtsinger_heldout_set'`.

- [ ] **Step 3: Implement the minimal builder**

```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path[:0] = [str(ROOT / "scripts"), str(CACHE / "soulx-singer")]
from prepare_diffsinger_gtsinger_ja import (  # noqa: E402
    DATASET, normalized_phones, selected_rows,
)
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402

INDICES = (165, 172, 174, 379, 380)

def build_ds_row(row: dict, audio_duration: float, f0: np.ndarray) -> dict:
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
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
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
            "id": identifier, "expected_text": ds["text"],
            "source_audio_path": str(source.relative_to(ROOT)),
            "ds_path": str(ds_path.relative_to(ROOT)),
            "alignment_status": "dataset_provided_manual",
            "target_f0_source": "rmvpe",
            "source_whisper_upper_bound": "must_be_recomputed_by_evaluator",
        })
    report = {
        "status": "evaluation_input_only", "rows": manifest,
        "license": "CC BY-NC-SA 4.0", "source_audio_committed": False,
        "release_allowed": False,
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and builder**

Run: `.venv-diffsinger/bin/pytest tests/test_diffsinger_heldout_gate.py::test_heldout_ds_row_preserves_manual_timing_and_rmvpe_grid -q`

Expected: `1 passed`.

Run: `.venv-diffsinger/bin/python scripts/build_diffsinger_gtsinger_heldout_set.py`

Expected: five `.ds` files, five manifest rows, all duration/F0 assertions passing, and no write under `data/source/`.

- [ ] **Step 5: Commit the builder and test**

```bash
git add scripts/build_diffsinger_gtsinger_heldout_set.py tests/test_diffsinger_heldout_gate.py
git commit -m "test(audio): add held-out singing set"
```

---

### Task 2: Multi-reference identity evidence in the existing evaluator

**Files:**
- Modify: `scripts/evaluate_diffsinger_pjs_rapid.py`
- Modify: `tests/test_diffsinger_heldout_gate.py`

**Interfaces:**
- Consumes: one or more `--identity-reference PATH` arguments.
- Produces: `similarity_summary(references: dict[str, tuple[np.ndarray, np.ndarray]], current: tuple[np.ndarray, np.ndarray]) -> dict` plus per-reference, mean, minimum, and maximum WavLM/ECAPA values in every candidate row.

- [ ] **Step 1: Write the failing summary test**

```python
from evaluate_diffsinger_pjs_rapid import similarity_summary

def test_identity_summary_keeps_every_reference_and_distribution():
    references = {
        "a.wav": (np.array([1., 0.]), np.array([1., 0.])),
        "b.wav": (np.array([0., 1.]), np.array([0., 1.])),
    }
    result = similarity_summary(references, (np.array([1., 0.]), np.array([1., 0.])))
    assert result["reference_count"] == 2
    assert result["wavlm"]["values"] == {"a.wav": 1.0, "b.wav": 0.0}
    assert result["wavlm"]["mean"] == .5
    assert result["ecapa"]["min"] == 0.0
```

- [ ] **Step 2: Run the test and confirm the missing function failure**

Run: `.venv-diffsinger/bin/pytest tests/test_diffsinger_heldout_gate.py::test_identity_summary_keeps_every_reference_and_distribution -q`

Expected: import failure for `similarity_summary`.

- [ ] **Step 3: Implement repeated references and the pure summary helper**

```python
def _stats(values: dict[str, float]) -> dict:
    data = list(values.values())
    return {"values": values, "mean": round(float(np.mean(data)), 5),
            "min": min(data), "max": max(data)}

def similarity_summary(references, current):
    wavlm = {name: round(float(np.dot(value[0], current[0])), 5)
             for name, value in references.items()}
    ecapa = {name: round(float(np.dot(value[1], current[1])), 5)
             for name, value in references.items()}
    return {"reference_count": len(references),
            "wavlm": _stats(wavlm), "ecapa": _stats(ecapa)}
```

Replace the argument and single-reference block with:

```python
parser.add_argument("--identity-reference", action="append", type=Path, default=[],
                    help="repeat for each real target-speaker reference")

if args.identity_reference:
    del whisper
    torch.cuda.empty_cache()
    feature_extractor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"),
        savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )
    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio16(path)
        values = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            first = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            second = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        first = torch.nn.functional.normalize(first, dim=-1).squeeze().cpu().numpy()
        second = second.squeeze().cpu().numpy()
        second /= max(np.linalg.norm(second), 1e-8)
        return first, second
    references = {}
    for raw_path in args.identity_reference:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        references[str(path.relative_to(ROOT))] = speaker(path)
    for row in rows:
        row["identity_similarity"] = similarity_summary(
            references, speaker(ROOT / row["audio_path"])
        )
```

Use both identity means only after `row["pass"]` in candidate ordering, and store the reference path list in the report. Do not convert identity into an automatic release claim.

- [ ] **Step 4: Run focused and existing evaluator tests**

Run: `.venv-diffsinger/bin/pytest tests/test_diffsinger_heldout_gate.py::test_identity_summary_keeps_every_reference_and_distribution tests/test_diffsinger_pjs.py::test_reference_relative_hf_spike_gate tests/test_diffsinger_pjs.py::test_equal_hop_pitch_gate_rejects_material_timing_drift -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the evaluator change**

```bash
git add scripts/evaluate_diffsinger_pjs_rapid.py tests/test_diffsinger_heldout_gate.py
git commit -m "test(audio): aggregate identity references"
```

---

### Task 3: Render and reject or promote from all five phrases

**Files:**
- Generate locally: `artifacts/reports/diffsinger_gtsinger_heldout_set/listening/`
- Generate: `artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_*.json`
- Modify: `docs/diffsinger_rapid_foundation.md`

**Interfaces:**
- Consumes: the five `.ds` rows and the existing soprano-15000, tenor-500, and tenor/GYU-mix20 diagnostic checkpoints.
- Produces: five per-phrase evaluation reports with free STT, F0, waveform, and five-reference identity distributions.

- [ ] **Step 1: Render identical conditions with official DiffSinger CLI**

For each manifest row, render depth `0`, seed `20260718` under:

```text
soprano: exp=gtsinger_ja_source, ckpt=15000, spk=gts_ja_soprano
tenor: exp=gtsinger_ja_tenor, ckpt=500, spk=gts_ja_tenor
mix20: exp=gtsinger_ja_tenor_gyu_identity, ckpt=100, spk=tenor:0.8|gyu:0.2
```

Expected: 15 WAVs; no spectral refiner is applied before the unmodified candidates pass lexical and waveform gates.

Run from the repository root:

```bash
mkdir -p artifacts/reports/diffsinger_gtsinger_heldout_set/listening
for ds in data/external/work/gtsinger/heldout_eval/*.ds; do
  id=$(basename "$ds" .ds)
  for spec in 'soprano|gtsinger_ja_source|15000|gts_ja_soprano' \
              'tenor|gtsinger_ja_tenor|500|gts_ja_tenor' \
              'mix20|gtsinger_ja_tenor_gyu_identity|100|tenor:0.8|gyu:0.2'; do
    IFS='|' read -r label exp ckpt spk_a spk_b <<< "$spec"
    spk="$spk_a${spk_b:+|$spk_b}"
    (cd data/cache/diffsinger && PYTHONPATH="$PWD/../../../scripts:$PWD" \
      ../../../.venv-diffsinger/bin/python scripts/infer.py acoustic "../../../$ds" \
      --exp "$exp" --ckpt "$ckpt" --spk "$spk" \
      --out ../../../artifacts/reports/diffsinger_gtsinger_heldout_set/listening \
      --title "${id}_${label}" --depth 0 --seed 20260718)
  done
done
```

- [ ] **Step 2: Evaluate each phrase against its source and all five GYU references**

Run the existing evaluator once per phrase with its manifest text/source/DS, all three candidate WAVs, and repeated references:

```bash
for id in gtsja0165 gtsja0172 gtsja0174 gtsja0379 gtsja0380; do
  expected=$(jq -r --arg id "$id" '.rows[] | select(.id==$id) | .expected_text' \
    artifacts/reports/diffsinger_gtsinger_heldout_set/manifest.json)
  source=$(jq -r --arg id "$id" '.rows[] | select(.id==$id) | .source_audio_path' \
    artifacts/reports/diffsinger_gtsinger_heldout_set/manifest.json)
  .venv-diffsinger/bin/python scripts/evaluate_diffsinger_pjs_rapid.py \
    --ds "data/external/work/gtsinger/heldout_eval/${id}.ds" \
    --expected-text "$expected" \
    --candidate "soprano=artifacts/reports/diffsinger_gtsinger_heldout_set/listening/${id}_soprano.wav" \
    --candidate "tenor=artifacts/reports/diffsinger_gtsinger_heldout_set/listening/${id}_tenor.wav" \
    --candidate "mix20=artifacts/reports/diffsinger_gtsinger_heldout_set/listening/${id}_mix20.wav" \
    --reference-audio "$source" \
    --identity-reference data/processed/master/212.wav \
    --identity-reference data/processed/master/215.wav \
    --identity-reference data/processed/master/216.wav \
    --identity-reference data/processed/master/219.wav \
    --identity-reference data/processed/master/220.wav \
    --output "artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_${id}.json"
done
```

Expected: each report contains a non-empty free transcript, SHA-256, pitch/voicing, waveform/spectral values, and `identity_similarity.reference_count == 5` for every candidate.

- [ ] **Step 3: Apply the release-independent gate without cherry-picking**

Reject any candidate if any of five phrases has Whisper similarity below `.8`, pitch p90 above `100` cents, gross pitch errors above `5%`, clipping, or reference-relative HF spike failure. Compare identity distributions only among audio-valid rows; require both WavLM and ECAPA means to improve over the matching unadapted foundation without a material per-reference collapse.

Expected: either one candidate is marked `human_listening_pending`, or all are explicitly rejected. Neither result authorizes RC8/RC9.

- [ ] **Step 4: Record the evidence and run repository checks**

Update `docs/diffsinger_rapid_foundation.md` with the five-phrase result and exact report paths.

Run: `.venv-diffsinger/bin/pytest tests/test_diffsinger_heldout_gate.py tests/test_diffsinger_pjs.py -q`

Expected: all DiffSinger PJS tests pass.

Run: `python scripts/validate_dataset.py`

Expected: `PASS recordings=132 sequential=106..237 pcm=48k_mono corrupt=0`.

- [ ] **Step 5: Commit only reproducible code, compact JSON evidence, and documentation**

```bash
git add docs/diffsinger_rapid_foundation.md artifacts/reports/diffsinger_gtsinger_heldout_set/*.json
git commit -m "test(audio): gate held-out singer quality"
```

Do not commit source/reference audio, rendered WAVs, external GTSinger files, model checkpoints, or copyrighted song material.
