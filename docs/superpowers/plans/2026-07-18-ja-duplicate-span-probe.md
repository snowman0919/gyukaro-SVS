# JA Duplicate-Span Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether a high-confidence CTC alignment can remove repeated Japanese OmniVoice content without changing the RC8 runtime.

**Architecture:** Add one pure function that validates CTC phone coverage, unknown ratio, monotonicity, and anchor confidence before producing a monotonic SoulX latent index map that skips only oversized unmatched source spans. A reproducible probe script compares the four approved paths, evaluates waveform/STFT, RMVPE, Whisper, and speaker metrics, and rejects the candidate unless every stated gate passes.

**Tech Stack:** Python, NumPy, PyTorch/torchaudio MMS-FA, SoulX, RMVPE, Whisper large-v3-turbo, WavLM, ECAPA, librosa.

## Global Constraints

- Diagnostic only: do not change the RC8 renderer or backend selection.
- Apply only to JA, non-rapid scores after high-confidence duplicate-span detection.
- Keep one phrase-level SoulX decode; no per-note TTS, WAV stitching, or waveform pitch shifting.
- Fall back completely when unknown ratio, target coverage, monotonicity, or span confidence fails.
- Preserve RC7 and all existing listening artifacts byte-for-byte.
- Do not start RC9/OpenUtau before RC8 human acceptance.

---

### Task 1: Pure duplicate-span latent map

**Files:**
- Modify: `src/gyu_singer/inference/content_timing.py`
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: `alignment: dict`, `source_duration: float`, `target_duration: float`, `frames: int`.
- Produces: `duplicate_span_content_warp(...) -> tuple[np.ndarray | None, dict]`; `None` means full fallback and the dict records the failed gate or removed spans.

- [x] **Step 1: Write failing tests**

Add synthetic phone alignments proving: a high-confidence oversized source gap is skipped monotonically; unknown-heavy input returns `None`; non-monotonic input returns `None`; normal aligned input returns `None` rather than inventing a duplicate.

- [x] **Step 2: Verify RED**

Run: `pytest -q tests/test_hybrid.py -k duplicate_span`

Expected: collection/import failure because `duplicate_span_content_warp` does not exist.

- [x] **Step 3: Implement the minimum function**

Use existing CTC phone rows. Reject unless known/high-confidence target coverage is at least `0.85`, unknown-phone ratio is at most `0.10`, target/source positions are monotonic, and an anchor pair identifies at least `0.50 s` excess source time at a source/target gap ratio of at least `2.5`. Return normalized, nondecreasing source positions and exact removed source intervals; do not edit existing `latent_content_warp`.

- [x] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_hybrid.py -k 'duplicate_span or ctc'`

Expected: all selected tests pass.

### Task 2: Reproducible four-path diagnostic

**Files:**
- Create: `scripts/probe_rc8_ja_duplicate_span.py`

**Interfaces:**
- Consumes: `examples/quality_ja.json`, `examples/heldout_ja.json`, fixed RC8 checkpoints/content settings, and the existing nine-file RC8 candidate manifest.
- Produces: `artifacts/reports/rc8_ja_duplicate_span/` containing WAVs, CTC evidence, waveform/STFT plots, `manifest.json`, and `evaluation.json`.

- [x] **Step 1: Render identical-condition paths**

For both JA scores, preserve fixed source/F0/identity/style/seed and render: current RC8, global `0.25` warp, existing grouped-content single-decode result, and duplicate-span candidate or its byte-identical fallback.

- [x] **Step 2: Record direct evidence**

Save OmniVoice and final Whisper transcripts, removed source intervals, target coverage, unknown ratio, monotonicity, span confidence, actual WAV paths, and SHA-256 values.

- [x] **Step 3: Measure required gates**

Reuse existing metric functions for RMVPE pitch MAE, voicing accuracy, HF spike, sample jump, WavLM/ECAPA similarity, and short/medium/long STFT analysis. Compare the full existing nine-file set; untouched non-JA files must retain SHA-256.

- [x] **Step 4: Decide without runtime integration**

Mark `diagnostic_reject` if held-out similarity is below `0.90`, repetition remains, quality JA regresses, pitch/voicing/artifact/identity gates regress, or any non-JA SHA changes. Otherwise mark only `diagnostic_candidate_human_pending`.

### Task 3: Verification and report

**Files:**
- Modify: `docs/rc8_quality_fixes.md`
- Modify: `docs/rc8_listening_report.md`

- [x] **Step 1: Run repository checks**

Run: `pytest -q tests/test_hybrid.py`, `python scripts/validate_dataset.py`, and the existing package smoke test command discovered from repository scripts/docs.

- [x] **Step 2: Verify scope**

Run `git diff --check`, confirm no RC7 hashes changed, and confirm no runtime renderer imports or calls the diagnostic function.

- [x] **Step 3: Report truthfully**

Document the four-path results, exact CTC evidence, waveform/STFT artifacts, WAV paths, and `human_pending`. If rejected, explicitly state that A was not integrated and only then begin duration-collapse dataset diagnosis in a later task.
