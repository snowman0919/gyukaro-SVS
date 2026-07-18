# GTSinger GYU Preservation-First Identity Adaptation Design

**Status:** Approved and executing as a bounded diagnostic
**Date:** 2026-07-18  
**Branch:** `codex/gtsinger-gyu-identity-design`  
**Base commit:** `20423ec3a48a55a1026ffa1237679a61cf7af073`

## 1. Purpose

This experiment asks one bounded question:

> Can a frozen, score-native GTSinger soprano foundation acquire measurable GYU identity without regressing lexical accuracy, score pitch, voicing, or waveform stability?

It does not change the production renderer. It does not create RC8, an OpenUtau package, or a release. The existing SoulX/OmniVoice experiments remain diagnostic evidence only and are not inputs to this path.

The experiment is deliberately ordered:

1. freeze and reproduce the accepted score-native acoustic foundation;
2. qualify Korean lexical control while the acoustic foundation is frozen;
3. train only a bounded, zero-initialized late identity adapter;
4. reject the whole candidate if any mandatory phrase-by-seed gate fails;
5. create a human A/B set only after all machine gates pass;
6. consider an RC8 candidate only after explicit human approval in a later task.

Lexical and pitch preservation are hard constraints. Identity improvement cannot compensate for a failed lyric, unstable seed, clipping, or an artifact regression.

## 2. Non-goals and prohibitions

This design does not authorize:

- changes to `main`, the production renderer, package code, or OpenUtau integration;
- creation of an RC, tag, package, or release;
- unfreezing or retraining the GTSinger acoustic foundation or vocoder;
- per-note TTS, waveform pitch shifting, final-WAV chunk stitching, or a source-loop renderer;
- stronger diffusion truncation, seed cherry-picking, or metric-based phrase selection;
- an acoustic refiner, spectral cleanup stage, or generic denoising workaround;
- use of real target RMVPE F0 as an inference or model input;
- modification or commit of original recordings under `data/source/`;
- commit of generated WAV files, checkpoints, external datasets, or caches.

GTSinger-derived checkpoints and outputs remain subject to the upstream CC BY-NC-SA 4.0 terms. Passing this diagnostic would not establish commercial or release compatibility.

## 3. Fixed foundation and evidence

### 3.1 Immutable model inputs

The experiment pins the following inputs before any result is observed:

| Component | Path or revision | SHA-256 / revision |
| --- | --- | --- |
| GTSinger JA soprano checkpoint | `data/cache/diffsinger/checkpoints/gtsinger_ja_source/model_ckpt_steps_15000.ckpt` | `dd31b42469ef2caa307799212b30fa44b2f1b7186c2f3a14eae45a2a80a6da8a` |
| Vocoder checkpoint | `data/external/work/diffsinger_score_native/vocoder_exported/model.ckpt` | `0b6728a7e677afdf0d1abc8d1fc1ac376631f6055062d2578db7d8ae4ba24729` |
| GTSinger dataset | upstream revision | `4426c862beed558b7e1cb8a4dce7e8c0c83bb208` |
| Official DiffSinger code | upstream revision | `0619d61d5301c4340db442a15cf3e73e197e9101` |

The effective dictionary, model configuration, phoneme inventory, sample rate, mel configuration, and inference settings must be fingerprinted alongside these revisions. A mismatch invalidates comparison results rather than silently creating a new foundation.

#### Execution provenance erratum

Before any new render, Git and GitHub both rejected `0619d61d5301c4340db442a15cf3e73e197e9101` as an unavailable OpenVPI DiffSinger commit. The local DiffSinger clone reflog shows it was cloned directly at `753b7cc622aadf802b3145d7bb8f7df4afa213c4`, and the frozen soprano checkpoint was created after that clone without an intervening checkout. The original protocol revision is therefore invalidated for incorrect provenance. Execution protocol revision 2 pins the evidenced code revision `753b7cc622aadf802b3145d7bb8f7df4afa213c4` before any new KO output is rendered; it retains the unavailable hash as an explicit reported-value error rather than silently replacing it.

### 3.2 Accepted score-native anchor

The authoritative foundation evidence is:

`artifacts/reports/diffsinger_gtsinger_heldout_set/aggregate_evaluation.json`

The JA soprano foundation passed all five independently aligned held-out phrases and all 15 fixed phrase-seed combinations for seeds 7, 21, and 42. Recorded aggregate values are:

| Metric | Value |
| --- | ---: |
| Lyric similarity mean | 0.92444 |
| Lyric similarity minimum | 0.83720 |
| Maximum pitch-error p90 | 33.12 cents |
| Minimum voicing accuracy | 0.8808 |
| Maximum HF spike / source | 1.0695 |
| Mean WavLM GYU similarity | 0.54093 |
| Mean ECAPA GYU similarity | 0.09710 |
| Phrase-seed passes | 15 / 15 |

The seed outputs have distinct byte hashes, so this gate measures seed stability rather than replaying one cached file. Older reports that describe an earlier rapid probe as `objective_rapid_gate: reject` are historical diagnostics; the later five-phrase aggregate above is the foundation anchor for this design.

## 4. Data boundaries and split policy

### 4.1 GYU adaptation corpus

The current phrase-chunk corpus is small and inferred:

- manifest: `data/manifests/diffsinger_gyu_phrase_chunks.jsonl`;
- 81 rows from 37 source rows;
- 67 train, 8 validation, and 6 test rows;
- approximately 4.742 minutes total;
- timing label: **inferred singing-aware CTC timing**;
- independently verified-score rows excluded from adaptation training.

These labels must remain described as inferred in every generated manifest and report. They are not manual score ground truth.

The train, validation, held-out, reference, phrase, and seed assignments are fixed before model results are inspected. Identical source recordings, lyric segments, or derived chunks must not cross splits.

### 4.2 Evaluation-only material

The following material is excluded from training and tuning:

- the five independently aligned GTSinger JA held-out phrases;
- Korean Rapid, Large Interval, Sustain, and Phrase Boundary stress phrases;
- all seed-7/21/42 evaluation renders;
- GYU reference recordings assigned to held-out identity measurement.

Rapid KO is protected regression evidence only. No failed phrase may be moved into training, and no reference or seed may be selected after metrics are known.

## 5. Two-stage adaptation

### 5.1 Stage A: Korean lexical qualification

The Japanese soprano acoustic model and vocoder remain frozen. Stage A may train only newly introduced Korean lexical parameters:

- `ko_*` phoneme/text embedding rows; and
- the smallest necessary lexical bridge that maps the existing Korean frontend inventory into the frozen encoder input space.

It must reuse the existing frontend and checkpoint plumbing wherever possible:

- `src/gyu_singer/frontend/phonemizer.py`;
- `scripts/prepare_diffsinger_gtsinger_gyu_identity.py`;
- `scripts/prepare_diffsinger_gyu_segments.py`;
- `scripts/prepare_diffsinger_gyu_phrase_chunks.py`.

The strict checkpoint whitelist remains the ownership boundary: only declared Korean lexical rows may change in Stage A. Speaker rows, encoder blocks, auxiliary decoder, diffusion backbone, variance predictors, and vocoder must be byte- or tensor-identical to the frozen foundation.

Stage A uses lexical/content supervision only. A frozen Korean CTC or content recognizer supplies the lexical loss. It must not use WavLM, ECAPA, GYU speaker centroid, speaker classification, or other identity objectives. This prevents Korean embeddings from becoming an uncontrolled timbre channel.

The four Korean stress cases are rendered from score-native `.ds` inputs rather than the SoulX runtime. Their notes, phonemes, durations, rests, tempo, and expected transcripts are recorded in a reproducible manifest. Existing example intent is sourced from:

- `examples/review_rapid_ko.json`;
- `examples/review_large_interval_ko.json`;
- `examples/review_sustain_ko.json`;
- `examples/review_phrase_boundary_ko.json`.

Stage A advances only if all four cases pass every fixed seed. Otherwise the lexical candidate is rejected and Stage B is not run.

### 5.2 Stage B: bounded late GYU identity adapter

Stage B starts from a Stage-A checkpoint that passed the lexical gate. It reuses the existing zero-initialized bounded `MelAdapter` in `scripts/train_diffsinger_gyu_mel_adapter.py` and the existing embedding path in `scripts/embed_diffsinger_mel_adapter.py`. The first experiment retains the existing 64-unit hidden layer, GELU activation, zero-initialized output projection, `tanh` residual bound, and 0.75 mel-unit per-bin limit. No new adapter family is introduced.

The adapter operates late in the phrase-level acoustic path, after score-native lexical and pitch structure has been established and before the frozen vocoder consumes the predicted mel representation. Its initial residual is zero, so adapter-ON and adapter-OFF mel and waveform tensors must match with `rtol=1e-5` and `atol=1e-6` under the same seed and deterministic settings.

Only adapter parameters are trainable. The foundation and vocoder parameters are frozen. If waveform-domain losses require decoding during training, vocoder operations remain in the computation graph so gradients reach the adapter, while vocoder parameter gradients must be absent.

The implementation must verify before optimization:

- initial adapter-ON and adapter-OFF output equivalence;
- finite, nonzero adapter gradients from the complete loss;
- no gradients on foundation or vocoder parameters;
- finite loss and audio tensors;
- bounded adapter gate, update norm, and output residual;
- reproducible inference for the fixed seed and input.

Failure of any feasibility check stops training.

## 6. Supervision and loss contract

### 6.1 Identity evidence

Identity supervision is computed from the actual final decoded waveform using frozen WavLM and ECAPA encoders. Each training or evaluation item is compared with a predeclared set of real GYU references and a reference centroid. Multi-reference results remain visible; a favorable centroid must not conceal a collapse against one reference.

The identity objective is a weighted combination of WavLM and ECAPA similarity losses. Neither metric may be optimized alone.

### 6.2 Preservation target

For the exact same score, phonemes, durations, and seed, the adapter-OFF Stage-A render is the preservation target. Stage B combines identity loss with:

- waveform preservation;
- multi-resolution STFT preservation at FFT sizes 256, 1024, and 4096;
- frozen content-feature preservation;
- voiced-frame pitch-period preservation;
- adapter output and gate regularization;
- parameter-delta and optimizer-update regularization relative to zero-initialized Stage B.

This is a constrained identity edit, not a reconstruction of the real recording. Score nominal F0 is the pitch condition. Real GYU audio supplies identity evidence only; its RMVPE target F0 never enters the candidate input.

The training log records every loss term plus adapter gradient norm, update norm, gate statistics, output residual norm, and `||delta theta|| / ||theta||`. Non-finite values or a configured bound violation stop the run and mark the checkpoint rejected.

## 7. Candidate comparison

The final comparison uses identical score, phonemes, durations, inference configuration, seed, loudness procedure, and metric implementations for:

1. frozen GTSinger soprano foundation;
2. Stage-A Korean lexical candidate;
3. Stage-B GYU identity candidate.

JA evaluation compares foundation and Stage B directly. KO evaluation uses the passed Stage-A candidate as the preservation baseline for Stage B, while also reporting the original foundation where the phoneme inventory permits a meaningful render.

Every JA held-out phrase and every KO stress case is rendered with seeds 7, 21, and 42. Reports contain per phrase, per seed, and per reference rows plus mean, median, minimum, standard deviation, and pass ratio. Aggregate means never override an individual mandatory failure.

## 8. Mandatory machine gates

The Stage-B candidate passes only if every applicable phrase-seed row passes all constraints below.

| Dimension | Mandatory gate |
| --- | --- |
| Lyric | Whisper lyric similarity at least 0.80 and no more than 0.02 below its preservation baseline; no repeated, substituted, or omitted phrase that changes the intended lyric |
| Pitch | RMVPE voiced-frame MAE no more than 10 cents above the preservation baseline; p90 no greater than 100 cents and no more than 15 cents above the preservation baseline |
| Voicing | Accuracy no more than 0.02 below the preservation baseline |
| Clipping | Zero clipped samples under the project threshold |
| HF stability | HF spike no greater than 1.10 times the preservation baseline |
| Sample continuity | Sample-jump p99.9 no greater than 1.10 times the preservation baseline |
| WavLM identity | Held-out mean at least 0.01 above the matched adapter-OFF baseline; no individual phrase/reference delta below -0.02 |
| ECAPA identity | Held-out mean at least 0.01 above the matched adapter-OFF baseline; no individual phrase/reference delta below -0.03 |
| Seed stability | Every phrase must pass seeds 7, 21, and 42; pass ratio must be 100% |

For JA, the matched adapter-OFF baseline is the frozen soprano foundation. For KO, it is the Stage-A lexical candidate with the identity adapter disabled. Identity deltas against the original unmodified foundation are also reported where its phoneme inventory permits a meaningful render, but they do not replace the matched-condition gate.

The same metric implementation, resampling rules, Whisper model/revision, RMVPE settings, WavLM/ECAPA revisions, GYU reference set, and thresholds are fingerprinted in the evaluation report.

Stage A uses the same lexical, pitch, voicing, clipping, HF, sample-jump, and seed-stability gates. Identity deltas are reported for observation but cannot make Stage A pass or fail.

## 9. Human A/B gate

Machine success creates only a `human_pending` diagnostic candidate.

The listening set is blind and loudness matched. It contains foundation/Stage-A versus Stage-B pairs for all four Korean stress phrases and representative JA held-out phrases, without best-sample selection. File naming and order are concealed in the listening copy while a separate manifest retains traceability.

Human review focuses on:

- lyric intelligibility and phoneme connections;
- note onset, duration, interval, and sustained-note stability;
- metallic, robotic, buzzing, tearing, and intermittent-noise artifacts;
- GYU identity improvement versus timbre drift;
- consistency across seeds.

Only the user’s explicit listening pass may authorize a later RC8-candidate task. Until then there is no renderer integration, package, OpenUtau work, tag, or release.

## 10. Failure handling

- If Stage A fails any Korean stress row, stop before identity training and record `lexical_reject`.
- If Stage B fails feasibility or any mandatory row, mark its checkpoint `diagnostic_reject` and do not connect it to a renderer or package.
- Preserve compact manifests, metric reports, logs, and selected listening evidence needed to explain the failure; keep large local WAV and checkpoint artifacts out of Git.
- Do not rescue a failed run by increasing model scope, selecting a favorable seed, weakening a gate, adding a spectral refiner, or applying waveform pitch shifting.
- A broader decoder/LoRA experiment requires a separate reviewed design.

## 11. Alternatives considered

### 11.1 Speaker-row interpolation or MixLN

This is rejected as the primary path. Earlier MixLN evidence preserved one rapid lyric but produced only small identity movement: WavLM approximately 0.63353 to 0.63382 and ECAPA 0.06300 to 0.07022. Five-phrase speaker mixing did not pass the foundation gate. Strict 5%, 10%, and 20% speaker-row blends also preserved a single phrase while WavLM remained about 0.51078 to 0.51231 and ECAPA declined from about 0.08363 to 0.08122.

### 11.2 Full or partial auxiliary-decoder tuning, LoRA, or broad fine-tuning

These offer more identity capacity but too much lexical/acoustic freedom for the first candidate. A prior tenor transfer changed roughly 16 million encoder/auxiliary parameters and failed lexical and HF gates. This option may be reconsidered only in a separate design if the bounded late adapter proves safe but insufficient.

### 11.3 Mel-L1 or envelope-only post adapters

Existing evidence does not show a joint identity gain. In the OpenUtau reassessment, the foundation WavLM/ECAPA values were approximately 0.5498/0.08918; the bounded mel adapter reached about 0.54765/0.09328, improving only one metric, while envelope adaptation reduced WavLM. Stage B therefore optimizes frozen final-WAV speaker evidence under strong preservation constraints instead of relying on average mel reconstruction.

## 12. Reproducibility and repository contract

Generated manifests and reports must be reproducible from scripts. A run record includes:

- Git commit and dirty-state check;
- all model, vocoder, dataset, dictionary, config, metric-model, and reference fingerprints;
- fixed split, phrase, and seed manifests;
- command lines and effective configuration;
- trainable and frozen parameter lists and counts;
- feasibility checks and training curves;
- complete per-row metrics and gate decisions;
- local artifact paths and checksums without committing large artifacts.

Any runtime change in a later, separately approved phase must run `python scripts/validate_dataset.py` and the package smoke test. This design phase itself must not modify runtime behavior.

## 13. Exit states

The experiment has only these valid outcomes:

1. **`lexical_reject`** — Korean score-native lexical qualification failed; identity training was not run.
2. **`diagnostic_reject`** — identity feasibility, preservation, identity gain, or any phrase-seed gate failed.
3. **`human_pending`** — every machine gate passed and a complete blind A/B set was produced.
4. **`human_passed_diagnostic`** — the user explicitly passed the A/B set; this permits a separate RC8-candidate design task but is not itself RC8 or a release.

No result from this design can directly be labeled production-ready, OpenUtau-ready, RC8, v1.0.0, or released.
