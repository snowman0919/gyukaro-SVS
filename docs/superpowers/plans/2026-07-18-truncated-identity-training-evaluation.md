# Truncated Identity Training and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train separate K=2 and K=4 v0.7 identity adapters and accept or reject them on every fixed held-out phrase × seed 7/21/42 combination.

**Architecture:** Capture the existing RC8 content/F0 inputs without changing the renderer, train only the existing identity FiLM parameters through the already-proven truncated 64-step/vocoder graph, then render four fixed identity conditions through one phrase-level SoulX decode. Run objective and free-Whisper analysis after generation; no candidate is connected to runtime.

**Tech Stack:** Python, PyTorch, existing GYU RC8 frontend, OmniVoice, SoulX-Singer, RMVPE, WavLM, SpeechBrain ECAPA, Whisper, pytest.

## Global Constraints

- Keep SoulX at 64 total phrase-level steps; train K=2 and K=4 separately.
- Freeze SoulX, vocoder, WavLM, ECAPA, content models, and style adapter.
- Train only `SoulXRealLatentAdapters.identity` from the current v0.7 checkpoint.
- Use fixed seeds `7`, `21`, and `42`; do not select phrases, seeds, references, or results after evaluation.
- Exclude `examples/heldout_ja.json` as a recorded content-source failure.
- Keep `examples/review_rapid_ko.json` evaluation-only.
- Do not modify renderer/runtime/package/checkpoints in place, `data/source/`, RC7 artifacts, RC8 outputs, RC9, or OpenUtau.
- Passing automated gates creates only `human_pending`; any failed gate is `diagnostic_reject`.

---

### Task 1: Freeze production content/F0 corpus

**Files:**
- Create: `scripts/prepare_truncated_identity_corpus.py`
- Test: `tests/test_truncated_identity_training.py`
- Runtime output: `artifacts/reports/truncated_identity_training/corpus/`

**Interfaces:**
- Produces: `fixed_split() -> dict[str, list[str]]`
- Produces: `data/manifests/truncated_identity_diagnostic.jsonl`

- [ ] **Step 1: Write failing split-policy tests**

Assert exact train, validation, held-out, excluded, and protected lists; assert no overlap; assert `heldout_ja` is excluded and `review_rapid_ko` is protected-only.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_truncated_identity_training.py`

Expected: FAIL because the corpus builder does not exist.

- [ ] **Step 3: Implement the minimal capture builder**

Instantiate `GyuSingerRC8Renderer` against the pinned main runtime, close only its SoulX worker, and replace that worker with a capture object. For each fixed score, let the existing renderer generate and adapt OmniVoice content, target F0, identity, content warp, CFG, and decoder options. The capture copies those inputs and raises a private completion signal before SoulX/refiners run. Record score SHA, source/F0 SHA, language, expected lyric, split, seed list, and `identity_training_eligible`.

Transcribe every source with free Whisper before training. A train/validation source below `0.80` lyric similarity, with a repeated expected span, or with a missing phrase stops the experiment. Held-out failures are retained and make promotion impossible rather than being silently removed.

- [ ] **Step 4: Run GREEN and build corpus**

Run:

```bash
pytest -q tests/test_truncated_identity_training.py
/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python scripts/prepare_truncated_identity_corpus.py
```

Expected: tests pass; 3 train, 3 validation, 5 held-out, 1 protected, and 1 excluded row are recorded.

- [ ] **Step 5: Commit corpus code and manifest metadata**

Stage only the script, test, JSONL manifest, and source-gate JSON. Do not stage generated source WAV/F0/warp files.

### Task 2: Train K=2 and K=4 candidates

**Files:**
- Create: `scripts/train_truncated_identity_final_wav.py`
- Modify: `tests/test_truncated_identity_training.py`
- Runtime output: `artifacts/reports/truncated_identity_training/k2/` and `k4/`

**Interfaces:**
- Consumes: the fixed corpus JSONL and helpers from `scripts/probe_truncated_identity_grad.py`.
- Produces: one experiment-only checkpoint and `training.json` per K.

- [ ] **Step 1: Write failing optimizer-safety tests**

Test that only identity parameters are returned, a frozen parameter gradient rejects the step, non-finite or zero gradients reject the step, and relative update/drift limits abort instead of saving a candidate.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_truncated_identity_training.py`

Expected: FAIL because training safety functions are absent.

- [ ] **Step 3: Implement bounded training**

Load one frozen SoulX/vocoder/WavLM/ECAPA context per process. Start from the v0.7 checkpoint and preserve a copy of every initial identity parameter. For each train phrase, use its complete audio when duration is 2–4 seconds; otherwise use the deterministic first voiced 3-second crop. Cycle seeds 7/21/42 in the frozen manifest order.

Run at most two epochs (`18` optimizer steps) with AdamW `lr=1e-4`, `weight_decay=0`, and gradient clipping at `0.1`. Use the approved speaker, waveform, FFT-256/1024/4096, content, pitch-period, adapter-output, gate, and parameter-drift losses. Record pre/post-clip gradient norm, update norm, and `||Δθ||/||θ₀||` every step. Abort before checkpoint save on non-finite loss/gradient, unexpected frozen gradient, gradient norm above `1.0`, per-step relative update above `0.005`, or total relative drift above `0.05`.

Evaluate the fixed validation phrases at seed 21 after each epoch. Select only between epoch 1 and epoch 2 by lower combined validation loss; held-out data is never loaded by the trainer. Save a full v0.7-compatible `SoulXRealLatentAdapters` state under an experiment-only path.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_truncated_identity_training.py`

Expected: all safety tests pass.

- [ ] **Step 5: Train K=2 and K=4 in fresh processes**

```bash
/home/kotori9/code/gyukaro/.venv-soulx/bin/python scripts/train_truncated_identity_final_wav.py --grad-steps 2 --output artifacts/reports/truncated_identity_training/k2
/home/kotori9/code/gyukaro/.venv-soulx/bin/python scripts/train_truncated_identity_final_wav.py --grad-steps 4 --output artifacts/reports/truncated_identity_training/k4
```

Expected: each run records at most 18 safe updates and writes a candidate checkpoint only when every safety gate passes.

### Task 3: Four-condition held-out gate

**Files:**
- Create: `scripts/evaluate_truncated_identity_candidates.py`
- Modify: `tests/test_truncated_identity_training.py`
- Runtime output: `artifacts/reports/truncated_identity_evaluation/`
- Modify: `docs/rc8_quality_fixes.md`

**Interfaces:**
- Consumes: current v0.7 plus K=2/K=4 experiment checkpoints and fixed corpus rows.
- Produces: `evaluation.json`, actual WAVs, per-case/seed waveform and MR-STFT plots.

- [ ] **Step 1: Write failing aggregation/gate tests**

Test mean, median, minimum, standard deviation, phrase×seed pass ratio, individual regression limits, Rapid KO protection, +0.01 mean gains over identity OFF, and at least +0.005 mean WavLM and ECAPA gains over current v0.7.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_truncated_identity_training.py`

Expected: FAIL because evaluator gates are absent.

- [ ] **Step 3: Render the fixed matrix**

Render identity OFF, current v0.7, K=2, and K=4 for all five held-out phrases plus protected Rapid KO at seeds 7/21/42. Hold source, target F0, reference, CFG, precision, and 64-step decode constant inside every comparison. Do not render excluded heldout JA.

- [ ] **Step 4: Analyze every WAV**

Run free Whisper, RMVPE, voicing accuracy, HF spike, sample jump p99.9, clipping, WavLM/ECAPA-to-fixed-GYU-centroid, waveform, and FFT-256/1024/4096 analysis for every output. Write sample-wise rows plus mean, median, minimum, standard deviation, and pass ratio.

- [ ] **Step 5: Apply mandatory gates**

Require the design thresholds against identity OFF and no meaningful individual regression. Additionally require candidate mean WavLM and ECAPA each exceed current v0.7 by at least `0.005`. A failed candidate is `diagnostic_reject`; a passing candidate is only `human_pending`.

- [ ] **Step 6: Verify and record**

```bash
pytest -q tests/test_truncated_identity_training.py
python scripts/validate_dataset.py
/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python -m pytest -q --ignore=tests/test_renderer.py
git diff --check
```

Add the measured result to `docs/rc8_quality_fixes.md`. Commit reproducible scripts/tests/manifests/JSON reports only; keep WAVs, plots, and experiment checkpoints uncommitted. Do not alter runtime or package files.
