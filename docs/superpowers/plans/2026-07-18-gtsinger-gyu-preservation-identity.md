# GTSinger-to-GYU Preservation Identity Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the pinned soprano foundation, reject before training if its Korean score-native 5×3 matrix fails, and run the approved bounded identity adapter only if that prerequisite passes.

**Architecture:** One diagnostic script reuses the existing phonemizer, phrase-frame builder, DiffSinger CLI, Whisper/RMVPE metrics, WavLM/ECAPA encoders, and acoustic metrics. It freezes inputs before rendering and writes compact evidence; large WAV/checkpoint/plot evidence stays under ignored `data/external/work/`. No runtime or package code is imported or modified.

**Tech Stack:** Python 3.11, PyTorch, OpenVPI DiffSinger, Transformers Whisper/WavLM, SpeechBrain ECAPA, SoulX RMVPE extractor, NumPy/SciPy, pytest.

## Global Constraints

- Use branch `codex/gtsinger-gyu-identity-design` at design commit `6d8f933f9a0087a8b4f0b4b742aca61aaad255c3`.
- Pin GTSinger soprano checkpoint SHA-256 `dd31b42469ef2caa307799212b30fa44b2f1b7186c2f3a14eae45a2a80a6da8a` and vocoder SHA-256 `0b6728a7e677afdf0d1abc8d1fc1ac376631f6055062d2578db7d8ae4ba24729`.
- Preserve the unavailable reported DiffSinger revision `0619d61d5301c4340db442a15cf3e73e197e9101` as an erratum and execute protocol revision 2 at the evidenced foundation-code revision `753b7cc622aadf802b3145d7bb8f7df4afa213c4`.
- Freeze cases `quality_ko`, `rapid_ko`, `large_interval_ko`, `sustain_ko`, and `phrase_boundary_ko`; freeze seeds 7, 21, and 42.
- Foundation KO failure stops before optimizer initialization with `foundation_ko_gate_reject` and final conclusion `diagnostic_reject`.
- Do not modify renderer, package, OpenUtau, main, RC7 evidence, source recordings, or rejected SoulX paths.
- Do not commit WAVs, checkpoints, caches, external datasets, plots, or copyrighted source audio.

---

### Task 1: Freeze and validate the protocol

**Files:**
- Create: `tests/test_gtsinger_gyu_identity_diagnostic.py`
- Create: `scripts/run_gtsinger_gyu_identity_diagnostic.py`
- Generate and commit: `data/manifests/gtsinger_gyu_identity_protocol.json`

**Interfaces:**
- Produces: `build_score_ds(score: dict) -> dict`, `distribution(values: list[float]) -> dict`, `gate_foundation(rows: list[dict], protocol: dict) -> dict`, and CLI subcommand `freeze`.
- Consumes: `phonemize`, `build_phrase_frames`, existing example scores, checkpoint/config/model paths, and fixed GYU references 212/215/216/219/220.

- [ ] **Step 1: Write failing tests**

Cover deterministic score-to-DS construction, five fixed cases, three fixed seeds, content hashes, no split/reference overlap, complete distribution statistics, missing/non-finite metric rejection, clipping rejection, and one-row lexical/seed failure rejecting the whole matrix.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python -m pytest tests/test_gtsinger_gyu_identity_diagnostic.py -q
```

Expected: collection fails because `run_gtsinger_gyu_identity_diagnostic` does not exist.

- [ ] **Step 3: Implement the minimum script**

The frozen thresholds are: lyric ≥0.80, pitch MAE absolute diagnostic plus p90 ≤100 cents, gross error ≤0.05, voicing ≥0.80, clipping 0, HF spike ≤2× the matching reference calibration when available, and complete 5×3 matrix. Every DS row uses inferred score-timed phoneme splits and declares that label.

- [ ] **Step 4: Verify GREEN and freeze**

Run the focused test, then:

```bash
/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python scripts/run_gtsinger_gyu_identity_diagnostic.py freeze
```

Expected: one immutable protocol manifest and five ignored `.ds` files; no source audio write.

- [ ] **Step 5: Commit**

Run `git diff --check` and commit tests, script, plan, and compact manifest as `test(audio): freeze preservation identity protocol`.

### Task 2: Reproduce the pinned Korean foundation gate

**Files:**
- Modify: `scripts/run_gtsinger_gyu_identity_diagnostic.py`
- Modify: `tests/test_gtsinger_gyu_identity_diagnostic.py`
- Generate locally: `data/external/work/gtsinger_gyu_identity_diagnostic/`
- Generate and commit: `artifacts/reports/gtsinger_gyu_identity_diagnostic/foundation_ko_evaluation.json`

**Interfaces:**
- Produces CLI subcommands `render-foundation` and `evaluate-foundation`.
- Reuses the combined deterministic vocabulary checkpoint only as an untrained Korean lexical initialization; the manifest records both its hash and the immutable soprano parent hash.

- [ ] **Step 1: Add failing matrix-completeness and gate tests**

Require exactly 15 unique `(case, seed)` rows, one existing WAV and SHA per row, finite required metrics, and rejection when any row fails.

- [ ] **Step 2: Verify RED**

Run the focused tests and confirm the new behavior is absent.

- [ ] **Step 3: Add minimum orchestration**

Use the unchanged cache checkout at the protocol-revision-2 DiffSinger commit, create an ignored inference experiment from the already-generated combined-vocabulary initialization, and call the official CLI at depth 0 for all 15 rows. Record command, runtime, SHA, and peak-memory evidence. Do not render a replacement after a failure.

- [ ] **Step 4: Evaluate every WAV once**

Use free Whisper with `language="ko"`, equal-hop RMVPE pitch/voicing, existing acoustic metrics, five fixed WavLM/ECAPA references, and reference audit fields. Write per-row and aggregate distributions. Generate local waveform and FFT 256/1024/4096 failure plots only after the machine decision.

- [ ] **Step 5: Apply the early-stop rule**

If any row fails, record `foundation_ko_gate_reject`, skip adapter feasibility/training, and continue directly to Task 4. If all pass, continue to Task 3 without changing the frozen protocol.

- [ ] **Step 6: Commit compact evidence**

Run focused tests and `git diff --check`; commit as `test(audio): evaluate korean score-native foundation`.

### Task 3: Conditional bounded identity adapter

**Condition:** Execute only when Task 2 reports all 15 Korean foundation rows passing.

**Files:**
- Modify: `scripts/run_gtsinger_gyu_identity_diagnostic.py`
- Modify: `tests/test_gtsinger_gyu_identity_diagnostic.py`
- Generate locally: adapter checkpoints, final WAVs, and plots under ignored external work.
- Generate and commit: compact feasibility, training, and JA/KO evaluation JSON.

**Interfaces:**
- Reuses the existing 64-hidden-unit, 0.75-limit, zero-output-projection `MelAdapter`.
- Trains only `fc1` and `fc2`; all DiffSinger and vocoder parameters remain frozen while vocoder operations retain the adapter gradient path.

- [ ] **Step 1: Write failing tests**

Cover initialization equivalence (`rtol=1e-5`, `atol=1e-6`), adapter-only `requires_grad`, gradient isolation, two-metric multi-reference aggregation, phrase×seed summary, preservation-before-identity checkpoint selection, inconsistent identity-gain rejection, and preservation-failure rejection.

- [ ] **Step 2: Verify RED**

Run focused tests and confirm each new assertion fails for missing behavior.

- [ ] **Step 3: Implement feasibility and fixed training**

Use only frozen-manifest training/validation rows. Log WavLM, ECAPA, frozen content, voiced pitch-period, waveform, FFT 256/1024/4096, adapter regularization, gradient/update norms, runtime, and peak memory. Stop on non-finite values, frozen gradients, zero identity gradient, clipping, repeated validation transcript, or preservation divergence.

- [ ] **Step 4: Select without held-out tuning**

Select lexicographically: all validation preservation gates, both identity improvements, smallest update, earliest checkpoint. Retain the fixed maximum checkpoint count from the protocol.

- [ ] **Step 5: Evaluate JA 5×3 and KO 5×3**

Require all individual rows to pass approved relative gates and held-out means to improve WavLM and ECAPA by at least 0.01. Create a blind loudness-matched A/B directory only when every gate passes; status is then `diagnostic_candidate_human_pending`.

- [ ] **Step 6: Commit compact evidence**

Run focused tests and `git diff --check`; commit as `test(audio): evaluate bounded identity adapter`.

### Task 4: Truthful diagnostic report and repository verification

**Files:**
- Create: `docs/final_gtsinger_gyu_identity_diagnostic.md`
- Generate and commit: `artifacts/reports/gtsinger_gyu_identity_diagnostic/final_status.json`

**Interfaces:**
- Consumes: frozen protocol, environment report, foundation result, and conditional adapter evidence.
- Produces exactly `diagnostic_reject` or `diagnostic_candidate_human_pending`.

- [ ] **Step 1: Generate the final report**

First line is `NOT A RELEASE REPORT — EXPERIMENT REJECTED` unless every machine gate passes, in which case it is `NOT A RELEASE REPORT — HUMAN LISTENING REQUIRED`. Include all required hashes, metrics, local evidence paths, skipped-phase reason, tests, dataset result, and explicit runtime/package/OpenUtau non-integration.

- [ ] **Step 2: Run complete verification**

```bash
git diff --check
/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python -m pytest -q --ignore=tests/test_renderer.py
/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python scripts/validate_dataset.py
```

Also verify no runtime/package/OpenUtau tracked diff, no committed WAV/checkpoint/cache/external data, no default checkpoint reference, and unchanged accepted/rejected status evidence.

- [ ] **Step 3: Commit the handoff**

Commit only compact evidence and report as `docs(audio): report gtsinger gyu identity diagnostic`. Leave branch and worktree intact; do not push, merge, or create a PR.
