NOT A RELEASE REPORT — MODEL, OPENUTAU RELEASE, AND PRODUCTION REMAIN BLOCKED

# GYU SVS Research Stack and Voicebank Factory Final Diagnostic Report

## Exact outcome

- Research/evidence normalization: complete
- Korean foundation: `foundation_machine_inconclusive`
- Korean representation selection: none
- Linguistic adapter: blocked, 0 optimizer steps
- Identity experiment: `identity_machine_inconclusive`, 0 optimizer steps, no runtime integration
- Runtime safety: complete; no production-approved backend
- Release gate: `release_blocked`, 11/11 gates failed
- OpenUtau release package: not created
- Voicebank factory infrastructure: complete
- Factory synthetic smoke: `dataset_needs_more_recording`
- Production readiness: FAIL
- Release readiness: FAIL

## Branch topology and commits

```text
1e72b4a  codex/gtsinger-gyu-identity-design
  └─ 16c6910 feat(research): normalize SVS evidence status
     └─ 18a085f docs(research): hand off SVS phase 0
        └─ 84b64d5 feat(experiment): add Korean phone gate
           └─ 5b96166 docs(experiment): record inconclusive gate
              └─ 60eca22 docs(experiment): hand off SVS phase 2
                 └─ 21fc40d feat(runtime): enforce experimental gates
                    └─ 163bd42 docs(runtime): hand off SVS phase 4
                       └─ 8680763 feat(release): fail closed on unsafe packages
                          └─ 6592dec docs(release): hand off SVS phase 5A
                             └─ cc9d1e5 feat(factory): add gated voicebank workflow
```

Branch heads at implementation completion:

- `codex/svs-01-research-evidence`: `18a085f8eef9d2577821406e94a7dc8107866035`
- `codex/svs-02-experiment-framework`: `60eca22021d5c00e808ac06e3f3095830fce8e94`
- `codex/svs-03-runtime`: `163bd4227a8788124b03fadbcf5d30fde9c13ec9`
- `codex/svs-04-release-openutau`: `6592decfdf29c3310c07e352ee412360e3ba21fe`
- `codex/svs-05-voicebank-factory` implementation: `cc9d1e588a1ed1f40d6cb9d4e04e1edf33d6ef86`

The final branch also contains documentation-only report/handoff commits after the implementation hash. No branch was merged, pushed, or opened as a PR.

## Verification by branch

| Branch | Full tests | Dataset | Specific result |
| --- | ---: | --- | --- |
| svs-01 | 141 passed | 132, corrupt 0 | 9-model status registry and evidence normalization |
| svs-02 | 162 passed | 132, corrupt 0 | Korean gate machine-inconclusive |
| svs-03 | 173 passed | 132, corrupt 0 | default runtime blocked; identity training false |
| svs-04 | 180 passed | 132, corrupt 0 | 11 release gates blocked; package refusal |
| svs-05 | 193 passed | 132, corrupt 0 | factory smoke needs more recording; wheel entrypoints pass |

All stacked ancestry checks passed. `git diff --check` passed on every milestone. The final wheel was built at `/tmp/gyukaro-wheel/gyu_singer-0.2.0-py3-none-any.whl` with SHA-256 `4f651d09a7a023e85341807b5d7a6f0523abe6abb1dac1b9d255bb75dc82c583`; both `gyu-voicebank --help` and installed `gyu-singer frontend` passed. This wheel is a package smoke artifact, not a release or voicebank.

## Environment

- Python 3.11.14 on Linux 6.17.0-1021-nvidia aarch64
- NVIDIA GB10, CUDA build 13.0, torch CUDA available, 128,452,014,080 GPU bytes total
- System memory 128,452,014,080 bytes; 121,726,607,360 available at audit
- Disk available 25,529,917,440 bytes; filesystem 98% used
- torch/torchaudio 2.11.0, numpy 2.3.5, scipy 1.17.1, soundfile 0.13.1
- transformers 5.8.1, onnxruntime 1.27.0, onnx 1.22.0
- h5py 3.16.0, praat-parselmouth 0.4.7, pyworld 0.3.5, pytest 9.1.0

`pip check` is not clean in the shared global environment. It reports unrelated pre-existing vLLM/rich/platform mismatches and a protobuf conflict between installed ONNX-era protobuf 6.33.6 and `descript-audiotools`' legacy `<3.20` requirement. Tests and the isolated wheel smoke passed, but future expensive model work should use a dedicated locked environment rather than this shared interpreter.

## Status registry

`configs/project_status.json` contains nine evidence-backed model rows and no production-approved pointer. RC7 is an accepted experimental baseline only. The SoulX phrase path, RC8 candidate 3, truncated K=2/K=4, GTSinger tenor, and GYU mix20 are rejected. GTSinger Japanese soprano is foundation-only. GTSinger Korean soprano is `foundation_machine_inconclusive`.

The CLI registry contains 14 implemented backends: 4 rejected and 10 blocked. Every one requires `--allow-experimental` and logs an `EXPERIMENTAL_OVERRIDE`. Omitting a backend for render/serve fails with a clear no-production message. Frontend-only use remains available.

## Korean lexical reassessment

The frozen probe covers ordinary, rapid, sustained, large-interval, phrase-boundary, single, repeated, coda, tense, aspirated, liaison, nasalization, and liquid-assimilation cases under seeds 7/21/42.

Three auditable representations were implemented:

- A `ko_components_v1`: explicit Hangul onset/nucleus/coda components
- B `ko_canonical_v1`: canonical Korean phones preserving tense and aspirated distinctions
- C `ko_onset_rhyme_v1`: onset plus rhyme with explicit coda state

Only A had existing rendered evidence; B and C were coverage-checked but not rendered. Therefore no representation was selected.

MMS forced alignment separated five script-matched real GYU references from mismatched controls: matched mean log score -1.340109, mismatched -5.330454, mean margin 3.990345, minimum margin 3.557688. Candidate case means ranged from -1.690442 to -4.592442. HuBERT cross-seed content consistency was 0.999795 across 15 files.

These values do not establish a pass. MMS is target-conditioned, and HuBERT consistency measures stability rather than phone correctness. No calibrated independent Korean singing phone recognizer was available. A threshold was not invented after observing the candidates. Whisper transcript fields are retained only as `auxiliary_stt_observation` with primary weight 0 and cannot alter the decision.

The alignment audit contains 149 inferred-only rows and 0 manual rows. The zero-equivalent linguistic adapter infrastructure exists, but representation selection is null, training status is `blocked_no_selected_representation`, optimizer steps are 0, and no identity objective was used.

## Identity and runtime

The bounded identity protocol freezes references 212/215/216/219/220 and seeds 7/21/42. Candidate order is fixed embedding, zero-equivalent FiLM, zero-equivalent low-rank residual, then disabled optional vocoder conditioning. The Korean foundation lacks phone validation and human approval, so training authorization is false. No checkpoint was trained or connected.

Runtime status is `runtime_safety_complete`. RC8 remains rejected, RC9 remains blocked, package/OpenUtau flags are false, and no experimental checkpoint is selected by default.

## Release and OpenUtau

The central engine requires approved foundation, approved identity, phone-centered lexical evidence, score/pitch, voicing, artifacts, multi-seed stability, long-form continuity, recorded human approval, license/provenance, and reproducible package evidence. Current result: 11 failed, `release_blocked`.

RC8 and RC9 normal packaging are refused before output creation. Diagnostic mode requires a `-diagnostic` name and emits metadata only with `NOT A RELEASE`; it bundles no checkpoints, source audio, samples, WAVs, or datasets. No OpenUtau library, release voicebank, tag, archive, or release was created.

## Voicebank factory

Commands:

```text
gyu-voicebank init --input DIR --name NAME --languages ko,ja,en --workspace DIR --rights-manifest FILE [--dry-run]
gyu-voicebank inspect --workspace DIR
gyu-voicebank prepare --workspace DIR [--review-manifest FILE]
gyu-voicebank train --workspace DIR
gyu-voicebank evaluate --workspace DIR
gyu-voicebank review-pack --workspace DIR
gyu-voicebank package --workspace DIR (--diagnostic | --release)
gyu-voicebank build --workspace DIR
gyu-voicebank status --workspace DIR
```

The factory requires affirmative rights/provenance, has no network acquisition, never overwrites sources, normalizes workspace copies, audits corruption/duplicates/acoustics, labels language estimates auxiliary, and excludes automatic Korean STT until user review. It produces energy-VAD segments, language phone sequences, inferred alignment confidence, correction hooks, coverage/recording plans, a non-SoulX adaptation plan, hashes, frozen splits, seeds, environment, early stop, and deterministic checkpoint selection.

The end-to-end smoke uses a generated 220 Hz sine wave, not a human voice or copyrighted material. It preserved the input SHA, normalized 44.1 kHz to 48 kHz mono, resumed idempotently, blocked training, refused release, and produced only diagnostic metadata. Result: `dataset_needs_more_recording`.

## Evidence and working state

Six compact tracked final reports are under:

- `artifacts/reports/korean_phone_reassessment/` (3 files)
- `artifacts/reports/runtime_safety/status.json`
- `artifacts/reports/release_gate/status.json`
- `artifacts/reports/voicebank_factory/smoke.json`

Local ignored Korean listening evidence contains 21 files, 7,115,352 bytes at `data/external/work/korean_phone_reassessment_review/`. The metadata-only RC8 package smoke contains 11 files, 2,347 bytes at `/tmp/gyukaro-rc8-diagnostic`. The wheel smoke contains one 81,344-byte wheel under `/tmp/gyukaro-wheel/`.

All five new stacked worktrees were clean after their implementation commits. The existing main worktree remains user-dirty with 385 status rows and was not altered or cleaned. The existing GTSinger design worktree is clean. The pre-existing held-out evidence worktree retains 67 local status rows and was not modified. Source recordings, checkpoints, caches, external datasets, and generated WAVs were not committed.

## Exact blockers and recommendation

1. No independent calibrated Korean singing phone recognizer or explicit human phone approval; Korean representation remains unselected.
2. No authorized Korean/multilingual acoustic foundation; linguistic and identity optimizer steps remain zero.
3. No production-approved backend.
4. All 11 release gates fail, including human approval and provenance/package evidence for a candidate.
5. The factory was validated only with a synthetic fixture; no new authorized real voicebank project met coverage.
6. Shared environment dependency conflicts and 98% disk use should be resolved before expensive work.

Recommended next action: keep all branches/worktrees as-is, obtain independent or human Korean phone validation, and run `gyu-voicebank init --dry-run` on an explicitly authorized real recording directory with a completed rights manifest. Do not train or package until coverage and foundation authorization pass.
