Overall status: primary `hybrid-svs` quality runtime passes the predeclared KO/EN/JA score, lyric, and held-vowel gate. Failed compact codec-latent checkpoint remains `hybrid-compact-experimental`.
Current stage: GYU Hybrid Singer v0.3 quality runtime; bootstrap-runnable, not an offline-weight bundle.
Package: `artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime.zip`
Package SHA-256: `aeabe177b535d1edbd5f8c02db17bfd973689b73ef707a0a05c069ab7a194795`
Git commit: current quality-controller source revision (recorded by Git)
Hybrid SVS checkpoint: `gyu_quality_pitch_controller.pt`, SHA-256 `c5aa5ef00101800d5d84cac453b8b0fad463567a5745bebc5933a4ab95d278f2`
Trainable parameters: 193,940 controller parameters.
Phrase-level neural generation: yes; full phrase controller, ACE-Step content, and SoulX neural decode.
Phoneme-note alignment: yes; deterministic language-aware mapping; real labels remain inferred.
Continuous pitch conditioning: yes; score F0/control in, flow-predicted expressive residual into SoulX; RMVPE F0 target only.
Teacher distillation gradient verified: yes; weighted teacher loss reaches controller timbre/language/style representations.
Pseudo-singing used in training: yes; 24 accepted compact rows and six explicitly synthetic quality-controller F0 rows.
Blurred boundary active: yes; quality controller condition path.
Conditional flow matching active: yes; controller source-to-residual flow before neural decode.
Style conditioning active: yes; controller controls plus ACE style prompt.
OpenUtau integration: executable USTX bridge through resident quality renderer; native registration not implemented.
Korean: quality gate pass.
English: quality gate pass.
Japanese: quality gate pass.

# What was inherited from Stage 0

132 indexed recordings, canonical 48 kHz masters, real phrase anchors, MOSS baseline, teacher manifests, and baseline resident runtime remain preserved. `baseline_renderer.py` and `neural_renderer.py` are comparison backends only.

# What was actually implemented in this Goal

`TriSingerModel(latent_dim=1)` now drives the primary quality path: unified phoneme, language, score, blurred-boundary, timbre, style, and pitch features flow through `ConditionalFlowTransformer` and `SingingDecoder` to produce an expressive F0 residual. Protocol v2, persistent ACE/SoulX workers, trilingual frontend, hard no-DSP path test, and executable USTX bridge are present.

# Exact hybrid forward path

Protocol v2 normalization converts beats to seconds. Frontend and deterministic note alignment build one full phrase tensor. `TriSingerModel.condition` fuses content, score, blurred boundaries, GYU reference timbre, style, and score-only continuous pitch. `SingingDecoder` creates a residual source and `ConditionalFlowTransformer` refines it. The bounded residual is added to the full 50 Hz nominal score contour, then ACE-Step generates one lyric phrase and SoulX-Singer decodes the complete phrase with that explicit F0. No primary call uses per-note TTS, pitch shift, phase vocoder, or waveform concatenation.

# Exact data flow by dataset type

Real GYU rows use inferred timing plus cached RMVPE F0, codec latents, and trust 1.0 in compact-model research. The deployed controller uses six explicitly synthetic ACE-Step/SoulX phrase rows with RMVPE F0 residual as target; it never receives target F0 as input. Teacher rows provide trust-weighted representation targets only, never acoustic or F0 targets.

# Teacher distillation path

665 weighted teacher rows pass through `acoustic_reference_features` and the deployed controller's `model.distillation_prediction`; `weighted_distillation_loss` consumes recorded `trust_weight` with coefficient 0.05. Gradient tests prove timbre, language, and style encoders receive non-zero teacher-loss gradients.

# Pseudo-singing generation and gates

ACE-Step-v1-3.5B (Apache-2.0) creates 100 KO/EN/JA a-cappella lyric candidates. Apache-2.0 SoulX SVC transfers GYU timbre. Gate requires RMVPE correlation ≥0.90, duration 0.85–1.15, WavLM ≥0.65, Whisper content ≥0.10, and matching transcript script-language. 27 rows pass; all rejections are evaluation-only. Provenance and metrics: `docs/pseudo_singing_report.md`.

# SVS concepts integrated

TCSinger-style blurred local context addresses hard boundaries; FM-Singer-style residual flow predicts expressive F0; TechSinger-style presets and continuous controls enter `StyleEncoder`. Exact forward calls, losses, tests, and metric status: `docs/component_traceability.md`.

# Gradient connectivity evidence

`tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` checks every retained model module. `test_teacher_distillation_reaches_timbre_language_and_style_encoders` checks teacher-stage gradients. Full suite result is recorded by `PYTHONPATH=src pytest -q`.

# Training runs

CUDA, AdamW `0.0002`, batch 1, 8000 steps; train rows 60 real + 24 pseudo; validation/test 8/5. Final logged losses: total 1.280458, flow 1.277616, pitch 0.00415, teacher 0.017563. `artifacts/reports/hybrid_training.json` records full history, validation losses, wall clock 302.948 s, GPU peak 207839232 bytes, and validation audio paths.

# Ablation results

`artifacts/reports/hybrid_teacher_ablation.json` contains short no-teacher versus weighted-teacher WavLM comparisons. It proves teacher branch changes output, not a quality gain.

# Baseline versus hybrid metrics

On identical four-note scores, primary `hybrid-svs` F0 correlation is KO
0.9942, EN 0.9604, JA 0.9905; pitch MAE is 11.08, 22.01, 13.40 cents; lyric
similarity is 1.0000, 0.7105, 0.6000. The old per-note DSP baseline reaches
0.2676/0.1944/0.4944 correlation and 0.0000/0.0526/0.0667 lyric similarity.
Exact evidence: `artifacts/reports/primary_vs_baseline_evaluation.json`.

# Listening sample paths

`artifacts/samples/baseline_{ko,en,ja}.wav`, `artifacts/samples/hybrid_{ko,en,ja}.wav`, and `artifacts/samples/ablation_{no_teacher,with_teacher}_{ko,en,ja}.wav`.

# OpenUtau integration status

`integrations/openutau/bridge.py` reads `.ustx`, selects voice part, converts project ticks and first tempo to protocol v2 seconds, writes JSON, and can POST to resident `/render`. `examples/openutau_smoke.ustx` has been exercised through the `hybrid-soulx-phrase` quality HTTP backend, producing a 48 kHz mono WAV. The service keeps one pinned ACE-Step worker and one pinned SoulX worker resident across requests. Tempo maps, native renderer registration, and editor curves are not implemented.

# Known failures

Generated compact-hybrid F0 does not reliably follow score; intelligibility is weak; 3-second vowels miss requested C4; real score labels are inferred; style controls uncalibrated; Korean-only real target data limits EN/JA evidence.  The v0.4 condition-source residual-flow sampler reduced latent validation loss but still fails generated quality (`artifacts/reports/hybrid_residual_flow_evaluation.json`).

# Primary quality-runtime evidence

Primary `hybrid-svs` loads the trained TriSinger pitch controller, which
predicts a bounded flow residual from score/content/timbre/style inputs. It
performs full-phrase ACE-Step lyric-vocal generation and full-phrase
SoulX-Singer neural timbre transfer conditioned on the resulting exact 50 Hz
score contour. It is not the source-loop renderer and it does not use per-note
TTS, pitch shifting, time stretching, or waveform concatenation.
Actual resident-runtime outputs passed the fixed objective gate: KO `0.9942` F0
correlation and `11.08` cents MAE; EN `0.9604` / `22.01`; JA `0.9905` /
`13.40`; held-note CV was at most `0.0049`; lyric similarity was
`1.0000/0.7105/0.6000`.
`artifacts/reports/soulx_runtime_smoke.json` is the exact evidence.

`artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime.zip` was unzipped
and its `run.sh` generated 48 kHz mono output. It includes the controller
checkpoint and a reproducible `bootstrap.sh` for the isolated pinned
Apache-2.0 ACE-Step/SoulX environments and model downloads. It is not
validation of the failed compact checkpoint.

# Claims that are explicitly not being made

No offline-weight bundle, compact-checkpoint quality claim, annotated source singing scores, native OpenUtau registration, or teacher-data quality-gain claim. The multilingual quality claim applies to the measured primary `hybrid-svs` runtime.

# Exact package commands

```sh
PYTHONPATH=src python scripts/package_quality_runtime.py
rm -rf /tmp/gyu-quality-smoke && mkdir /tmp/gyu-quality-smoke
unzip -q artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime.zip -d /tmp/gyu-quality-smoke
cd /tmp/gyu-quality-smoke/gyu-hybrid-singer-v0.3-quality-runtime
sh bootstrap.sh /path/to/cache
GYU_SINGER_CACHE=/path/to/cache GYU_SOULX_PYTHON=/path/to/cache/soulx-singer/.venv/bin/python sh run.sh
```

# Highest-value next steps

Add legal scored singing supervision with real EN/JA target data, train duration residuals and contour losses, capture held-out validation/telemetry, then rerun multi-seed listening and objective tests before reassessing v1.
