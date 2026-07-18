NOT A RELEASE REPORT — EXPERIMENT REJECTED

# GTSinger-to-GYU preservation identity diagnostic

## Decision

- Conclusion: `diagnostic_reject`
- Foundation gate: `foundation_ko_gate_reject` (0/15, pass ratio 0.000)
- Training: `not_started_foundation_gate_failed`
- Failure taxonomy: `foundation_content_failure`
- Selected checkpoint: none
- Human A/B: not generated
- Runtime integration: false
- Package/OpenUtau: blocked and unchanged
- RC9, production, and v1.0.0 remain prohibited.

The frozen Korean foundation emitted repeated or substituted syllables on every phrase×seed item. Mandatory lexical qualification therefore failed before optimizer initialization. Identity scores cannot compensate for this failure, and the approved protocol forbids attempting to repair it with the adapter.

## Authority and provenance

- Specification: `docs/superpowers/specs/2026-07-18-gtsinger-gyu-preservation-identity-design.md`
- Approved design commit: `6d8f933f9a0087a8b4f0b4b742aca61aaad255c3`
- Starting commit: `6d8f933f9a0087a8b4f0b4b742aca61aaad255c3`
- Protocol revision: 2; invalidated revision: 1
- Protocol manifest: `data/manifests/gtsinger_gyu_identity_protocol.json` (`f138edfc8270dbc9a04043ff83d40f06ab101b3798608e7b41b6c88c9aa93996`)
- DiffSinger actual cached revision: `753b7cc622aadf802b3145d7bb8f7df4afa213c4`
- Reported but unavailable DiffSinger revision: `0619d61d5301c4340db442a15cf3e73e197e9101`; retained as a provenance erratum and not used.
- GTSinger dataset revision: `4426c862beed558b7e1cb8a4dce7e8c0c83bb208`
- Foundation checkpoint SHA-256: `dd31b42469ef2caa307799212b30fa44b2f1b7186c2f3a14eae45a2a80a6da8a`
- Combined-vocabulary zero-step initialization SHA-256: `f7ab385ccf2b4cb7f16973d177e3d1924b840971fe241df9e6dec3cebf4b9916`
- Vocoder SHA-256: `0b6728a7e677afdf0d1abc8d1fc1ac376631f6055062d2578db7d8ae4ba24729`
- RMVPE SHA-256: `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193`

Implementation commits before this report:

- `46d2fdb0136307ddc9e1026b4faa7c7895b8cc32 test(audio): freeze preservation identity protocol`
- `2b48a653df5ac25613f8d23a36c1410d92754067 fix(audio): correct DiffSinger protocol provenance`
- `28d3bc268e37713c61cb4f0bfd72c4122ab03528 test(audio): evaluate korean score-native foundation`

## Environment

- Python 3.11.14; PyTorch 2.11.0+cu130; CUDA build 13.0
- GPU: NVIDIA GB10; unified memory 128452014080 bytes
- System memory: 128452014080 bytes
- Disk at freeze: 27616337920 free of 982819848192 bytes
- Libraries: torch 2.11.0, torchaudio 2.11.0, numpy 2.3.5, scipy 1.17.1, librosa 0.11.0, transformers 5.8.1, speechbrain 1.1.0, onnxruntime 1.27.0, lightning 2.3.3, pytorch-lightning 2.6.5
- Peak GPU memory: unavailable (`nvidia-smi` reports N/A for this GB10 unified-memory system); per-render peak process RSS and runtime are retained sample-wise.

## Frozen protocol

- Cases: ordinary `quality_ko`, rapid `rapid_ko`, large interval `large_interval_ko`, sustain `sustain_ko`, phrase boundary `phrase_boundary_ko`
- Seeds: [7, 21, 42]
- Identity references: 212, 215, 216, 219, 220; one fixed set for every sample
- Adapter (authorized but not instantiated): {"type": "MelAdapter", "hidden": 64, "limit": 0.75, "initial_output_projection": "zero"}
- Optimizer (frozen but not initialized): {"type": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001, "maximum_steps": 200, "evaluation_interval": 25, "maximum_checkpoints": 3, "early_stop_failed_intervals": 3}
- Loss weights (frozen but not evaluated): {"wavlm_identity": 0.5, "ecapa_identity": 0.5, "content": 2.0, "pitch_period": 2.0, "waveform": 1.0, "stft_256": 1.0, "stft_1024": 1.0, "stft_4096": 1.0, "adapter_delta": 0.1, "parameter_update": 1.0}
- Adaptation split sizes: train 67, validation 8, held-out 6; no split leakage.
- All phoneme splits are marked inferred; score timing and nominal F0 are not relabeled as GYU supervision.

### Fixed reference audit

- `data/processed/master/212.wav` — `b7e2e58b63d6bd9dd54f26063ed7fd16ef06ce14c497f1edf620fc3480b799c4`; 48000 Hz, 1 ch, clip=0.0, transcript=`아침이 오면 닫힌 창문 너머로 조용히 번지는 빛을 따라 걸어가 아직은 서툰 마음이라 해도 멈추지 않고 다시 노래할 거야`
- `data/processed/master/215.wav` — `889758edb40fbba14a8e157598d608c5db486f4f44460ef56528e7e8e5a2992f`; 48000 Hz, 1 ch, clip=0.0, transcript=`멀어진 계절의 끝에서 잊혀진 이름을 다시 불러 흩어진 바람 사이로 작은 목소리가 피어나`
- `data/processed/master/216.wav` — `106b008c13abccc2a7f09b3e665fb7cbe7fbf4486acffee9f6a80e40656642c3`; 48000 Hz, 1 ch, clip=0.0, transcript=`푸른 밤하늘 아래서 나는 나의 길을 찾아가 흔들리는 시간 속에도 이 노래만은 남아있어 가장 낮은 마음의 자리에서 다시 작은 숨을 고르고 아직 보이지 않는 내일을 향해 천천히 걸어가`
- `data/processed/master/219.wav` — `83d922254988d5f31b0a485a27dc13300d53d613daae6297bb66fa586bed0c49`; 48000 Hz, 1 ch, clip=0.0, transcript=`잠시 멈춰 서서 숨을 고르면 멀리 사라진 빛도 다시 보여`
- `data/processed/master/220.wav` — `7dca1d4c4e6146326e9507461f8160f42a4746d0a967ef1e19f7cd6cbaa48e69`; 48000 Hz, 1 ch, clip=0.0, transcript=`오늘의 하늘은 맑아 가벼운 바람이 불어 닫힌 문을 열고서 다시 앞으로 가 손끝에 닿은 햇살 마음에 번지는 노래 넘어져도 괜찮아 다시 시작하면 돼`

No source recording, external dataset, rendered WAV, plot, cache, or checkpoint is committed. The audit found no clipping or duplicate reference transcript; no favorable per-phrase reference selection occurred.

## Parameter and feasibility audit

- Executed foundation state dict: 210 tensors, 27,274,368 values.
- Trainable parameters in the executed experiment: 0 (0%).
- The authorized identity adapter was deliberately not instantiated because the pre-optimizer foundation gate failed.
- Consequently initialization equivalence, adapter gradient isolation, update norms, adapter VRAM, and optimizer feasibility are not applicable—not passed.
- Foundation, variance/pitch/duration predictors, phoneme encoder, decoder, and vocoder were not modified.

## Training and checkpoint selection

- Optimizer steps: 0 of the frozen maximum 200.
- Training/validation/held-out checkpoint selection was never entered.
- Selected checkpoint: none.
- No held-out-guided tuning, seed selection, reference selection, or loss-weight change occurred.

## Korean qualification results (5×3)

| Case | Seed | Whisper | Lyric | Repeat | Pitch MAE | Pitch p90 | Voicing | Clip | HF spike | Jump | WavLM | ECAPA | WAV path | SHA-256 |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| quality_ko | 7 | 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 | 0.000000 | True | 4.168300 | 2.394000 | 0.989960 | 0.000000 | 36.486300 | 0.064232 | 0.492268 | 0.130009 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/quality_ko_foundation_seed7.wav` | `0dcf76fceb93e412fef7dbbcf758a77dccbdb9ec466514cbeeaa70ef04cc9182` |
| quality_ko | 21 | 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 | 0.000000 | True | 4.095400 | 2.367500 | 0.989960 | 0.000000 | 36.822400 | 0.064163 | 0.492653 | 0.130154 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/quality_ko_foundation_seed21.wav` | `34df1868406ac5fa219b9be8ec075aee68f6bb22ebca8a1eddd038b3402272fb` |
| quality_ko | 42 | 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 와우 | 0.000000 | True | 4.158300 | 2.341600 | 0.989960 | 0.000000 | 36.668800 | 0.064154 | 0.483707 | 0.129693 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/quality_ko_foundation_seed42.wav` | `424afd69848764ab89b6f0bbc09978a52fd4489ebb86d24b781e4e563b83bd9d` |
| rapid_ko | 7 | 야다 야다 야다 야다 | 0.000000 | True | 6.143300 | 13.045200 | 0.980000 | 0.000000 | 15.732800 | 0.057935 | 0.594446 | 0.217715 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/rapid_ko_foundation_seed7.wav` | `e0ccc95dc40157f014998fdfc133eb9423cb3808f80d956c8f332eac9b9aed3f` |
| rapid_ko | 21 | 야다 야다 야다 야다 | 0.000000 | True | 6.412500 | 13.268700 | 0.980000 | 0.000000 | 15.631500 | 0.058087 | 0.597636 | 0.219090 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/rapid_ko_foundation_seed21.wav` | `299f7f055bcd42c15f7c95783ce71e2f60d1759e5e5b245ee23d4967676a84b2` |
| rapid_ko | 42 | 야다 야다 야다 야다 | 0.000000 | True | 6.532500 | 13.080000 | 0.980000 | 0.000000 | 15.734900 | 0.057992 | 0.593603 | 0.217755 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/rapid_ko_foundation_seed42.wav` | `af7a8c0ce76b250ea1df7f5ad9115ba4d6275373dcf4eb9e76e45d00b69b6d18` |
| large_interval_ko | 7 | 아 아 | 0.333333 | True | 10.809800 | 5.800500 | 0.958333 | 0.000000 | 38.396200 | 0.085692 | 0.451421 | 0.092253 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/large_interval_ko_foundation_seed7.wav` | `ef23f371cbb9c72848806738404ef577754afb7803b260cf1662ebc22fa568df` |
| large_interval_ko | 21 | 아 아 | 0.333333 | True | 10.837800 | 5.670000 | 0.958333 | 0.000000 | 38.269300 | 0.085592 | 0.455439 | 0.093131 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/large_interval_ko_foundation_seed21.wav` | `799a62c9e505c95074f54d963679c1ea6bdb9c20ba74214c802ca3d55a9f1740` |
| large_interval_ko | 42 | 아 아 | 0.333333 | True | 10.828200 | 5.924000 | 0.958333 | 0.000000 | 38.689700 | 0.085765 | 0.462258 | 0.090675 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/large_interval_ko_foundation_seed42.wav` | `4869d4ad6dd7928fa734646ae48be914e1ff02b8d81f4e265077dbc00468d0b8` |
| sustain_ko | 7 | 다 | 0.000000 | False | 1.428700 | 1.811800 | 0.990000 | 0.000000 | 16.667100 | 0.043985 | 0.329727 | 0.046802 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/sustain_ko_foundation_seed7.wav` | `7e55ccfcfcf90f47c5fbd15f7f0231b9a036f99e57ba0f72810a3ed4884fe894` |
| sustain_ko | 21 | 다 | 0.000000 | False | 1.418100 | 1.843200 | 0.990000 | 0.000000 | 16.440000 | 0.043981 | 0.329162 | 0.047049 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/sustain_ko_foundation_seed21.wav` | `cc28a09de2dd7e4f9f01b263d7ccc36729f04e74f4e1889267f6da1db0bfbc8e` |
| sustain_ko | 42 | 다 | 0.000000 | False | 1.428800 | 1.863100 | 0.990000 | 0.000000 | 16.680000 | 0.043987 | 0.329654 | 0.046995 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/sustain_ko_foundation_seed42.wav` | `561080d99776915b64833697f7d13fb98278e7bf6981168ec08e8e93d8eb06c6` |
| phrase_boundary_ko | 7 | 다아... 다아... | 0.000000 | True | 3.188800 | 7.142200 | 0.918919 | 0.000000 | 11.609300 | 0.035422 | 0.471492 | 0.248109 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/phrase_boundary_ko_foundation_seed7.wav` | `ae6ebe9433047c335629a174240f08d2f4038c2a7cb3fd66b71cb755ddf674a1` |
| phrase_boundary_ko | 21 | 다아... 다아... | 0.000000 | True | 10.475800 | 8.500600 | 0.864865 | 0.000000 | 11.160700 | 0.035220 | 0.461869 | 0.241955 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/phrase_boundary_ko_foundation_seed21.wav` | `c35726c7c8f1e6fe719a2825d2f341a0ca81e1aa779826ceca4efb78d0ad5a12` |
| phrase_boundary_ko | 42 | 다아... 다아... | 0.000000 | True | 3.130600 | 7.337300 | 0.864865 | 0.000000 | 11.378500 | 0.035249 | 0.462776 | 0.251662 | `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/phrase_boundary_ko_foundation_seed42.wav` | `c0797fe649f4465b238db378abc73e8fcdc073198f0587362fd12f6ad4f60e93` |

Aggregate distributions:

| Metric | Mean | Median | Minimum | Maximum | Std |
|---|---:|---:|---:|---:|---:|
| lyric_similarity | 0.066667 | 0.000000 | 0.000000 | 0.333333 | 0.133333 |
| pitch_mae_cents | 5.670460 | 4.168300 | 1.418100 | 10.837800 | 3.445048 |
| pitch_p90_abs_cents | 6.159313 | 5.800500 | 1.811800 | 13.268700 | 4.108780 |
| gross_error_over_600_cents | 0.001325 | 0.000000 | 0.000000 | 0.006667 | 0.002255 |
| voicing_accuracy | 0.960235 | 0.980000 | 0.864865 | 0.990000 | 0.041944 |
| clip_fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| hf_spike_p99_over_median | 23.757833 | 16.667100 | 11.160700 | 38.689700 | 11.417730 |
| sample_jump_p999 | 0.057430 | 0.057992 | 0.035220 | 0.085765 | 0.017406 |
| spectral_flux_p95 | 0.196165 | 0.227112 | 0.012387 | 0.351179 | 0.108327 |

Observed lexical failures were stable across seeds: ordinary produced repeated `와우`; rapid produced repeated `야다`; large interval produced `아 아`; sustain produced `다`; phrase boundary produced repeated `다아`. Pitch, voicing, clipping, and artifact metrics passing on some rows do not override the 0/15 lexical result.

## Japanese held-out and identity-candidate evaluation

The previously verified unadapted soprano Japanese gate remains the external foundation reference (5/5 phrases and 15/15 seed matrix) in `artifacts/reports/diffsinger_gtsinger_heldout_set/aggregate_evaluation.json`. No adapted candidate exists, so a new candidate Japanese 5×3 matrix, identity-gain comparison, protected regression matrix, or human A/B set would be misleading and was not generated after the mandatory Korean early-stop.

WavLM and ECAPA were nevertheless recorded for every Korean foundation WAV against every fixed reference to make the rejection auditable. The compact evaluation contains each sample's per-reference values and mean/median/minimum/maximum/standard deviation. They are baseline observations, not identity improvements: no trained condition exists from which to compute a gain.

## Machine gates

- Korean lexical validity all phrases/seeds: FAIL (0/15)
- No repetition/omission/substitution: FAIL
- Seed stability: FAIL at the lexical level across all seeds
- Pitch/voicing/clipping/artifact preservation: measured, but cannot promote a lexically failed foundation
- WavLM and ECAPA held-out improvement: NOT EVALUATED; no candidate
- Individual regression limits: NOT EVALUATED; no candidate
- Overall mandatory gate: FAIL

Failure taxonomy is `foundation_content_failure`, not `adapter_content_regression`: the failure occurred before an adapter or optimizer existed.

## Evidence

- Compact evaluation: `artifacts/reports/gtsinger_gyu_identity_diagnostic/foundation_ko_evaluation.json`
- Local WAVs: `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/`
- Local render logs: `data/external/work/gtsinger_gyu_identity_diagnostic/logs/`
- Local waveform + FFT 256/1024/4096 plots: `data/external/work/gtsinger_gyu_identity_diagnostic/failure_plots/`
- Local DS inputs: `data/external/work/gtsinger_gyu_identity_diagnostic/inputs/`
- Local evidence: 66 files, 24290488 bytes; ignored and uncommitted.
- Every WAV path and SHA-256 is recorded in the sample table and compact JSON.

## Repository verification

- Relevant/full test suite: 133 passed.
- Dataset validation: `PASS recordings=132 sequential=106..237 pcm=48k_mono corrupt=0`.
- `git diff --check`: PASS before report commit and required again after commit.
- Production renderer imports no experimental adapter; package config selects no checkpoint; OpenUtau paths are unchanged.
- Previous RC7 and rejected SoulX evidence are unchanged. RC7 remains an accepted experimental baseline; RC8 candidate 3, v0.7 adapter, truncated K=2, and truncated K=4 remain rejected.

## Final conclusion

`diagnostic_reject`

The bounded experiment completed at its mandatory early-stop. The unadapted Korean score-native lexical foundation is not qualified, so GTSinger-to-GYU identity adaptation, checkpoint creation, runtime integration, packaging, OpenUtau work, RC9, and release work remain blocked.
