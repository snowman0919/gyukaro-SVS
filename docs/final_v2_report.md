Overall status: two distinct experimental paths exist.  The compact GYU
checkpoint fails quality and is not v1.  The dynamic frozen ACE-Step ->
SoulX-Singer phrase backend passes the predeclared KO/EN/JA score, hold, and
ASR gate, but its package depends on an external 13 GB model cache and pinned
SoulX runtime.  Therefore this is **not** a self-contained v1 release.
Current stage: compact `GYU Hybrid Singer v0.2-experimental` plus
`hybrid-soulx-phrase` quality-runtime proof.
Package: `artifacts/package/gyu-hybrid-singer-v0.2-experimental.zip`
Package SHA-256: `965491a14a4068dc58156275f780b60159f1f9c22b02841b8e4ab64d2be2be6a`
Git commit: `0a8fdde` (quality-runtime and reconstructed-score implementation revision)
Hybrid SVS checkpoint: `checkpoints/gyu_hybrid_v0.2.pt`, SHA-256 `788dbd03b3755aa324cec88813ceb214ce81d9f77d94cbf8e06a0f3b1f71d184`
Trainable parameters: 762,210.
Phrase-level neural generation: yes; one full phrase frame tensor, conditional-flow latent sample, frozen codec decode.
Phoneme-note alignment: yes; deterministic language-aware note-frame mapping; real/pseudo bootstrap labels marked inferred.
Continuous pitch conditioning: yes; MIDI, curve/residual, UV mask, masked log-F0 loss; current generated F0 quality fails.
Teacher distillation gradient verified: yes; timbre, language, and style encoder non-zero gradients under teacher loss.
Pseudo-singing used in training: yes; 24 accepted low-trust train rows, 100 candidates/27 accepted overall.
Blurred boundary active: yes; `BlurredBoundaryEncoder` is in condition path.
Conditional flow matching active: yes; rectified-flow target/noise objective and Euler phrase sampling.
Style conditioning active: yes; preset plus five controls; not calibrated.
OpenUtau integration: executable USTX parser/exporter plus resident renderer POST; native renderer registration not implemented.
Korean: runtime/frontend exercised; quality not accepted.
English: runtime/frontend exercised; quality not accepted.
Japanese: runtime/frontend exercised; quality not accepted.

# What was inherited from Stage 0

132 indexed recordings, canonical 48 kHz masters, real phrase anchors, MOSS baseline, teacher manifests, and baseline resident runtime remain preserved. `baseline_renderer.py` and `neural_renderer.py` are comparison backends only.

# What was actually implemented in this Goal

`TriSingerModel` connects unified phoneme, language, score, blurred-boundary, timbre, style, pitch, conditional-flow, and acoustic-latent decoder modules. Protocol v2, a resident hybrid renderer, trilingual frontend, hard no-DSP path test, and executable USTX bridge are present.

# Exact hybrid forward path

Protocol v2 normalization converts beats to seconds. Frontend and deterministic note alignment build one full phrase frame tensor. `TriSingerModel.condition` fuses content, score, blurred boundaries, reference timbre, style, and continuous pitch. `ConditionalFlowTransformer` samples one latent phrase; `SingingDecoder` adds learned acoustic bias; frozen `MossCodecDecoder` emits 48 kHz WAV. No hybrid call uses per-note TTS, pitch shift, or phase vocoder.

# Exact data flow by dataset type

Real GYU rows use inferred timing plus cached RMVPE F0, codec latents, and trust 1.0. ACE-Step lyric-vocal then SoulX SVC rows use 48 kHz synthetic audio, inferred RMVPE pitch/duration labels, and trust 0.20 only after hard gates. Teacher rows provide representation targets only, never codec acoustic targets.

# Teacher distillation path

665 weighted teacher rows pass through `acoustic_reference_features` and `model.distillation_prediction`; `weighted_distillation_loss` consumes recorded `trust_weight` with coefficient 0.15. Gradient tests prove timbre, language, and style encoders receive non-zero teacher-loss gradients.

# Pseudo-singing generation and gates

ACE-Step-v1-3.5B (Apache-2.0) creates 100 KO/EN/JA a-cappella lyric candidates. Apache-2.0 SoulX SVC transfers GYU timbre. Gate requires RMVPE correlation ≥0.90, duration 0.85–1.15, WavLM ≥0.65, Whisper content ≥0.10, and matching transcript script-language. 27 rows pass; all rejections are evaluation-only. Provenance and metrics: `docs/pseudo_singing_report.md`.

# SVS concepts integrated

TCSinger-style blurred local context addresses hard boundaries; FM-Singer-style rectified flow generates codec latents; TechSinger-style presets and continuous controls enter `StyleEncoder`. Exact forward calls, losses, tests, and metric status: `docs/component_traceability.md`.

# Gradient connectivity evidence

`tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` checks every retained model module. `test_teacher_distillation_reaches_timbre_language_and_style_encoders` checks teacher-stage gradients. Full suite result is recorded by `PYTHONPATH=src pytest -q`.

# Training runs

CUDA, AdamW `0.0002`, batch 1, 8000 steps; train rows 60 real + 24 pseudo; validation/test 8/5. Final logged losses: total 1.280458, flow 1.277616, pitch 0.00415, teacher 0.017563. `artifacts/reports/hybrid_training.json` records full history, validation losses, wall clock 302.948 s, GPU peak 207839232 bytes, and validation audio paths.

# Ablation results

`artifacts/reports/hybrid_teacher_ablation.json` contains short no-teacher versus weighted-teacher WavLM comparisons. It proves teacher branch changes output, not a quality gain.

# Baseline versus hybrid metrics

Current hybrid F0 correlation: KO 0.3390, EN -0.2012, JA 0.1278. ASR similarity: KO 0.0000, EN 0.0455, JA 0.0000. Full identical-score comparison: `docs/evaluation_v2_report.md`.

# Listening sample paths

`artifacts/samples/baseline_{ko,en,ja}.wav`, `artifacts/samples/hybrid_{ko,en,ja}.wav`, and `artifacts/samples/ablation_{no_teacher,with_teacher}_{ko,en,ja}.wav`.

# OpenUtau integration status

`integrations/openutau/bridge.py` reads `.ustx`, selects voice part, converts project ticks and first tempo to protocol v2 seconds, writes JSON, and can POST to resident `/render`. `examples/openutau_smoke.ustx` has been exercised through the `hybrid-soulx-phrase` quality HTTP backend, producing a 48 kHz mono WAV. The HTTP service is resident but invokes pinned ACE-Step/SoulX workers per render. Tempo maps, native renderer registration, and editor curves are not implemented.

# Known failures

Generated compact-hybrid F0 does not reliably follow score; intelligibility is weak; 3-second vowels miss requested C4; real score labels are inferred; style controls uncalibrated; Korean-only real target data limits EN/JA evidence.  The v0.4 condition-source residual-flow sampler reduced latent validation loss but still fails generated quality (`artifacts/reports/hybrid_residual_flow_evaluation.json`).

# Quality-runtime evidence (separate from the failed compact checkpoint)

`hybrid-soulx-phrase` performs full-phrase ACE-Step lyric-vocal generation
and full-phrase SoulX-Singer neural timbre transfer conditioned on the exact
50 Hz score F0 contour.  It is not the source-loop renderer and it does not
use per-note TTS, pitch shifting, time stretching, or waveform concatenation.
Actual CLI outputs passed the fixed objective gate: KO `0.9942` F0 correlation
and `11.08` cents MAE; EN `0.9637` / `21.52`; JA `0.9917` / `12.63`; held-note
CV was at most `0.0056`; lyric similarity was `1.0000/0.7105/0.6000`.
`artifacts/reports/soulx_runtime_smoke.json` is the exact evidence.

`artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime.zip` was unzipped
and its `run.sh` generated 48 kHz mono output.  It is intentionally a thin
runtime package: it requires external cached Apache-2.0 ACE-Step and
SoulX-Singer weights plus a compatible pinned SoulX environment.  Do not
describe it as standalone, production-ready, or as validation of the failed
compact checkpoint.

# Claims that are explicitly not being made

No standalone v1 release, production singer, compact-checkpoint quality claim, annotated source singing scores, or teacher-data quality gain. The multilingual quality claim applies only to the measured external-cache `hybrid-soulx-phrase` runtime.

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
