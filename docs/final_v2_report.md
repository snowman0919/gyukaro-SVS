Overall status: experimental neural phrase SVS; quality gate fail, not v1.
Current stage: GYU Hybrid Singer v0.2-experimental.
Package: `artifacts/package/gyu-hybrid-singer-v0.2-experimental.zip`
Package SHA-256: `72ff1dbc8d322875b5bcfb7cad964124ce5fd85b1329aac2f4285a10cd642a49`
Git commit: `7c4276851b1f20fb9eaef62c7e3a4654b300eb0e` (report generation source revision)
Hybrid SVS checkpoint: `checkpoints/gyu_hybrid_v0.2.pt`, SHA-256 `73d457dfb9e9267d06e600fe31bbea2198d68af4e95c9779dd9441c2851f0595`
Trainable parameters: 762,018.
Phrase-level neural generation: yes; one full phrase frame tensor, conditional-flow latent sample, frozen codec decode.
Phoneme-note alignment: yes; deterministic language-aware note-frame mapping; real/pseudo bootstrap labels marked inferred.
Continuous pitch conditioning: yes; MIDI, curve/residual, UV mask, masked log-F0 loss; current generated F0 quality fails.
Teacher distillation gradient verified: yes; timbre and language encoder non-zero gradients under teacher loss.
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

665 weighted teacher rows pass through `acoustic_reference_features` and `model.distillation_prediction`; `weighted_distillation_loss` consumes recorded `trust_weight` with coefficient 0.15. Gradient tests prove timbre and language encoders receive non-zero teacher-loss gradients.

# Pseudo-singing generation and gates

ACE-Step-v1-3.5B (Apache-2.0) creates 100 KO/EN/JA a-cappella lyric candidates. Apache-2.0 SoulX SVC transfers GYU timbre. Gate requires RMVPE correlation ≥0.90, duration 0.85–1.15, WavLM ≥0.65, Whisper content ≥0.10, and matching transcript script-language. 27 rows pass; all rejections are evaluation-only. Provenance and metrics: `docs/pseudo_singing_report.md`.

# SVS concepts integrated

TCSinger-style blurred local context addresses hard boundaries; FM-Singer-style rectified flow generates codec latents; TechSinger-style presets and continuous controls enter `StyleEncoder`. Exact forward calls, losses, tests, and metric status: `docs/component_traceability.md`.

# Gradient connectivity evidence

`tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` checks every retained model module. `test_teacher_distillation_reaches_timbre_and_language_encoders` checks teacher-stage gradients. Full suite result recorded in this worktree: 22 passed.

# Training runs

CUDA, AdamW `2e-4`, batch 1, 1,200 steps; train rows 60 real + 24 pseudo; validation/test 8/5. Final logged losses: total 1.324734, flow 1.321476, pitch 0.018544, teacher 0.015539. `artifacts/reports/hybrid_training.json` is full per-step evidence. GPU peak memory and wall-clock were not captured.

# Ablation results

`artifacts/reports/hybrid_teacher_ablation.json` contains short no-teacher versus weighted-teacher WavLM comparisons. It proves teacher branch changes output, not a quality gain.

# Baseline versus hybrid metrics

Current hybrid F0 correlation: KO 0.6464, EN 0.3920, JA -0.1025. ASR similarity: KO 0.0000, EN 0.0000, JA 0.2000. Full identical-score comparison: `docs/evaluation_v2_report.md`.

# Listening sample paths

`artifacts/samples/baseline_{ko,en,ja}.wav`, `artifacts/samples/hybrid_{ko,en,ja}.wav`, and `artifacts/samples/ablation_{no_teacher,with_teacher}_{ko,en,ja}.wav`.

# OpenUtau integration status

`integrations/openutau/bridge.py` reads `.ustx`, selects voice part, converts project ticks and first tempo to protocol v2 seconds, writes JSON, and can POST to resident `/render`. Tempo maps, native renderer registration, and editor curves are not implemented.

# Known failures

Generated hybrid F0 does not reliably follow score; intelligibility is weak; 3-second vowels are voiced but miss requested C4; real score labels are inferred; style controls uncalibrated; Korean-only real target data limits EN/JA evidence.

# Claims that are explicitly not being made

No v1 release, production singer, quality superiority over baseline, Japanese singing-quality support, annotated source singing scores, or teacher-data quality gain.

# Exact package commands

```sh
PYTHONPATH=src python scripts/package_v1.py
rm -rf /tmp/gyu-hybrid-smoke && mkdir /tmp/gyu-hybrid-smoke
unzip -q artifacts/package/gyu-hybrid-singer-v0.2-experimental.zip -d /tmp/gyu-hybrid-smoke
cd /tmp/gyu-hybrid-smoke/gyu-hybrid-singer-v0.2-experimental
PYTHONPATH=runtime python -m gyu_singer.cli --backend hybrid-svs --checkpoint model/gyu_hybrid_v0.2.pt --audio-tokenizer model/moss-audio-tokenizer-nano --reference model/gyu_reference_216.wav render examples/smoke.json --output output.wav
```

# Highest-value next steps

Add legal scored singing supervision with real EN/JA target data, train duration residuals and contour losses, capture held-out validation/telemetry, then rerun multi-seed listening and objective tests before reassessing v1.
