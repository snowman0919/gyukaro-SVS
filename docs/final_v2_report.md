Overall status: experimental hybrid neural SVS runtime verified; quality gate not passed.
Current stage: GYU Hybrid Singer v0.2-experimental.
Package: `artifacts/package/gyu-hybrid-singer-v0.2-experimental.zip`.
Package SHA-256: `02aec3b64907f0acd60fe250dbe4771c6b1d280b5bab82f0458f13fcdd57117f`.
Git commit: `9976681931b50f5b48d468a8ce2d98a380be68cb`.
Hybrid SVS checkpoint: `checkpoints/gyu_hybrid_v0.2.pt`, SHA-256 `8768102c2f61be037b32e60a7e1af2cf3b704c8dc13cb8d50d7a0f4cfbfd1ac4`.
Trainable parameters: 762,018.
Phrase-level neural generation: yes; full phrase frame tensor, one CFM latent sample, frozen codec decode.
Phoneme-note alignment: yes; deterministic explicit frame mapping, real-anchor scores inferred.
Continuous pitch conditioning: yes; nominal MIDI, RMVPE/inferred F0, UV mask, residual, masked log-F0 loss.
Teacher distillation gradient verified: yes; timbre and language encoder gradients nonzero in teacher-loss test.
Pseudo-singing used in training: no; Fish S2 pilot rows excluded after license review.
Blurred boundary active: yes; `BlurredBoundaryEncoder` in condition path.
Conditional flow matching active: yes; rectified-flow latent velocity training and Euler inference.
Style conditioning active: yes; preset plus five controls in `StyleEncoder`; calibration not established.
OpenUtau integration: executable USTX bridge/export plus resident render POST; no native engine registration.
Korean: runtime exercised; quality not accepted.
English: runtime/frontend exercised; quality not accepted.
Japanese: runtime/frontend exercised; no Japanese support claim.

## What was inherited from Stage 0

132 source recordings, canonical 48 kHz masters, real anchor supervision, 64 MOSS SFT pairs, 900 teacher generations, 633 weighted teacher rows plus 32 style rows, SoulX pilot, resident renderer, and baseline artifacts remain preserved. `baseline_renderer.py` and `neural_renderer.py` remain comparison renderers only. Historical package is explicitly relabeled `GYU Neural Vocalizer Baseline v0.1`; it is not v1 SVS.

## What was actually implemented in this Goal

`TriSingerModel` has connected `UnifiedPhonemeEncoder`, `LanguageFeatureEncoder`, `ScoreEncoder`, `BlurredBoundaryEncoder`, `TimbreEncoder`, `StyleEncoder`, `PitchConditionEncoder`, `ConditionalFlowTransformer`, and `SingingDecoder`. Protocol v2 supports beat timing, curves, style preset, and deterministic frame conversion. Hybrid resident service exposes `/health`, `/model`, and `/render`. USTX bridge exports score v2 and can return renderer WAV.

## Exact hybrid forward path

`normalize_score` validates score v2. `phonemize` makes language-aware units. `build_phrase_frames` creates one phrase's phoneme IDs, language features, note index/onset/duration/boundary, nominal or curve F0, UV, and residual. `TriSingerModel.condition` fuses content, score, blurred context, reference timbre, style, and pitch. `ConditionalFlowTransformer` predicts latent velocity during rectified-flow training and Euler sampling. `SingingDecoder` adds latent bias. Frozen `MossCodecDecoder` decodes one 768-D phrase latent stream to WAV. `hybrid.py` contains no `pitch_shift`, `phase_vocoder`, note loop, or `NeuralRenderer` call.

## Exact data flow by dataset type

Real GYU: inferred note timing plus cached RMVPE F0, codec latent target, trust 1.0 CFM/pitch loss. Fish S2 `[singing]` then SoulX SVC pilot WAVs are evaluation-only because Fish's license forbids the relevant generative-model training use. Teacher rows: teacher audio feature target only; no codec latent and no hard GYU-singing acoustic supervision. Train/validation/test contain 60/5/5 real rows.

## Teacher distillation path

Both required teacher manifests total 665 rows. Each teacher audio becomes `acoustic_reference_features`; `model.distillation_prediction` is compared by trust-weighted representation loss. Loss coefficient is 0.15. `test_losses_use_pitch_mask_and_teacher_trust` verifies zero trust does not contribute. `test_teacher_distillation_reaches_timbre_and_language_encoders` verifies intended gradients.

## Pseudo-singing generation and gates

Three controlled Fish S2 `[singing]` to SoulX SVC probes were measured: KO RMVPE 0.9918/WavLM 0.8585/ECAPA 0.7218, EN 0.9778/0.9632/0.6769, JA 0.9964/0.8812/0.7253. Candidate rows record duration ratio, ASR content, language, speaker scores, provenance, and synthetic status. They are rejected from training under Fish license terms. A SoulX-direct or otherwise explicitly permitted pipeline is required. Corpus has 0 legally admitted items, below requested 100–500.

## SVS concepts integrated

TCSinger-style soft context: 5-frame blurred boundary convolution. FM-Singer-style CFM: interpolated noise/codec target velocity MSE plus sampled Euler path. TechSinger-style technique conditioning: 8 presets and 5 continuous controls. Traceability, forward sites, losses, tests, and current evaluation state are in `docs/component_traceability.md`.

## Gradient connectivity evidence

`tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` passes for all retained model components. Teacher test passes nonzero timbre and language gradients. Phrase sample, protocol, resident HTTP, frontend KO/EN/JA, USTX conversion, no-DSP regression, alignment, flow loss, pitch, and trust-loss tests all pass. Full suite: 22 passed.

## Training runs

Main run: CUDA GB10, AdamW LR 2e-4, batch 1, 1,200 steps, 762,018 trainable parameters. Last metrics: total 1.503340, flow 1.488786, pitch 0.018468, teacher 0.090877. Full history: `artifacts/reports/hybrid_training.json`. GPU peak memory, wall clock, and held-out objective are missing and not claimed.

## Ablation results

Compatible 80-step checkpoints compare teacher coefficient 0 against 0.15. WavLM cosine to real GYU: no teacher KO/EN/JA `0.5288/0.5582/0.5165`; teacher `0.5310/0.5429/0.5169`. Teacher branch is connected and changes output, but this short A/B does not prove overall gain.

## Baseline versus hybrid metrics

Evidence: `artifacts/reports/baseline_hybrid_evaluation.json`, RMVPE from SoulX. Hybrid current KO/EN/JA F0 correlation is `-0.1638/-0.3146/-0.4096`, pitch MAE `1812.01/1886.69/1819.80` cents, and ASR similarity `0.0000/0.1250/0.1818`. Boundary-energy jump is lower for KO and JA but worse for EN in this seed. These results fail v1 pitch and intelligibility bars.

## Listening sample paths

`artifacts/samples/baseline_ko.wav`, `baseline_en.wav`, `baseline_ja.wav`, `hybrid_ko.wav`, `hybrid_en.wav`, `hybrid_ja.wav`; A/B samples are `artifacts/samples/ablation_{no_teacher,with_teacher}_{ko,en,ja}.wav`. They are evaluation artifacts, not quality endorsements.

## OpenUtau integration status

`integrations/openutau/bridge.py` parses USTX YAML voice parts, maps ticks using project resolution and first tempo, writes protocol v2, and POSTs resident `/render` when requested. Unit test proves tick conversion. Native OpenUtau renderer registration, tempo maps, and editor curves remain blocked/unimplemented; bridge is a real exporter, not a native plugin claim.

## Known failures

Hybrid generated F0 does not follow score; intelligibility is weak; real scores are inferred from speech timing; no legally admitted pseudo corpus exists; EN/JA have no real GYU singing supervision; style controls are uncalibrated; full staged validation and training telemetry are absent; evaluation has short synthetic phrases and baseline generation variance.

## Claims that are explicitly not being made

No v1 release. No production singer. No neural quality superiority over Stage 0. No Japanese singing support based only on WAV output. No claim teacher data improves quality. No claim inferred annotations are source singing scores. No claim baseline DSP is primary SVS.

## Exact package commands

```sh
PYTHONPATH=src python scripts/package_v1.py
rm -rf /tmp/gyu-hybrid-smoke && mkdir /tmp/gyu-hybrid-smoke
unzip -q artifacts/package/gyu-hybrid-singer-v0.2-experimental.zip -d /tmp/gyu-hybrid-smoke
cd /tmp/gyu-hybrid-smoke/gyu-hybrid-singer-v0.2-experimental
sh install.sh
sh run.sh
```

Package smoke already passed with system dependencies: 48 kHz mono, 307,200 frames.

## Highest-value next steps

Generate and quality-gate 100–500 legally usable scored pseudo phrases across stated registers and techniques; replace inferred real scores with annotated singing; train with real phrase validation/early stopping; add duration residual learning; measure listening tests and stable reproducible multi-seed metrics; then reassess v1 only after pitch, intelligibility, and timbre pass.
