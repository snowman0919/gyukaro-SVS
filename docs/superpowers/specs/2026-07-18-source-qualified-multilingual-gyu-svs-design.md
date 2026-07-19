# Source-Qualified Multilingual GYU SVS Design

## Status and scope

This design is approved for implementation as a local, non-commercial experiment. It does not authorize a public model release. GTSinger-derived data and checkpoints remain subject to CC BY-NC-SA 4.0, and source audio stays outside Git.

The objective is a genuinely usable GYU AI singing voice, not another renderer that merely emits WAV files. Completion requires a trained score-native model, stable Korean/English/Japanese rendering, human listening approval, and a DiffSinger-compatible OpenUtau package that renders in a clean local environment.

Existing conclusions remain unchanged:

- RC7 is an accepted experimental baseline only.
- RC8 candidate 3, the v0.7 SoulX identity adapter, and truncated K=2/K=4 are rejected.
- SoulX, RVC, Seed-VC, waveform pitch shifting, per-note TTS, and final-WAV stitching are not reopened.
- No experimental checkpoint is connected to the production renderer before all gates pass.

## Evidence behind the design

The repository contains 132 unmodified GYU recordings. Twenty-four phrases have independently verified scores. Seventy-six additional phrases have inferred RMVPE/script-constrained scores and must remain labeled inferred. Prior direct DiffSinger, speech-prior, MLP-Singer identity, post-source conversion, and latent-adapter experiments preserved either lyrics or identity, but not both.

The GTSinger Korean metadata contains 2,295 rows: 257 KO-Soprano-1, 759 KO-Soprano-2, and 1,279 KO-Tenor-1. The full Korean tree is about 5.44 GiB, while the two soprano subsets are about 2.35 GiB. A deterministic source probe found:

| Source | Rows | Whisper similarity >= 0.80 | Mean similarity | Minimum |
|---|---:|---:|---:|---:|
| KO-Soprano-1 | 11 | 0 | 0.4596 | 0.0000 |
| KO-Soprano-2 | 10 | 4 | 0.6752 | 0.0667 |

Some source segments contain lyrics absent from their metadata. Therefore the complete Korean corpus is not valid supervision. Only a source-qualified subset may enter training.

## Chosen architecture

The experiment uses the existing pinned OpenVPI DiffSinger training and export stack. It does not add a new synthesis runtime.

```text
qualified KO/EN/JA score-aligned singing
                 |
                 v
multilingual, multi-speaker DiffSinger foundation
  - language-tagged phonemes
  - score and explicit F0 conditioning
  - speaker conditioning learned from multiple singers
                 |
       freeze lexical, timing, pitch, decoder backbone
                 |
                 v
real GYU verified/inferred score pairs
  - new GYU speaker embedding
  - bounded speaker-conditioning adapter
                 |
                 v
KO/EN/JA stress and identity gates
                 |
                 v
human listening approval
                 |
                 v
DiffSinger/OpenUtau local package
```

The key difference from the rejected single-speaker adapter experiments is that speaker conditioning is learned as part of a multi-speaker score-native foundation before GYU adaptation. Language and singer are separate conditioning axes. The same GYU identity condition is used for KO, EN, and JA; EN/JA prosody remains generic because no real GYU EN/JA singing exists.

## Phase 1: freeze the source qualification protocol

Pin the GTSinger dataset revision `4426c862beed558b7e1cb8a4dce7e8c0c83bb208`, OpenVPI DiffSinger revision, vocoder revision, evaluator revisions, seeds, manifests, thresholds, and split policy before training.

Start with KO-Soprano-2 Control_Group rows. Do not download or preprocess the tenor subset. KO-Soprano-1 is excluded from the first foundation because its deterministic source probe passed zero of eleven rows.

A source row is accepted only when all of these are true:

- source WAV exists, is mono or deterministically downmixed, and is decoded without error;
- duration is 2 to 15 seconds;
- phone, phone-duration, note-pitch, and note-duration arrays exist and have equal lengths;
- absolute difference between summed phone duration and WAV duration is at most 50 ms;
- no clipping samples are present;
- normalized free-Whisper lyric similarity is at least 0.80;
- MMS-CTC target-phone coverage is at least 0.90;
- CTC unknown-phone ratio is at most 0.05;
- CTC alignment is monotonic and contains no unsupported repeated span;
- the row contains at least one voiced note and one lexical phone;
- all inferred or corrected fields are labeled with their actual source.

Whisper is a required gate, not the sole alignment authority. CTC, waveform duration, score arrays, and source listening evidence must agree.

Training may begin only if the accepted Korean subset has at least 200 unique rows, 30 minutes of singing, 40 fast rows, 20 high-register rows, 20 sustained-note rows, and 20 rows containing intervals of at least seven semitones. Split by complete song so no song or paired technique rendition crosses train, validation, and held-out sets.

If this gate fails, record `foundation_source_gate_reject`. Do not lower thresholds after seeing the result. The next valid path is a new rights-controlled GYU score-native recording campaign.

## Phase 2: Korean score-native foundation gate

Train a bounded Korean foundation first. Use the existing DiffSinger acoustic configuration, 48 kHz output, explicit score/F0 conditioning, and the existing pinned vocoder. Do not add post-processing.

The maximum run is 15,000 acoustic steps with fixed evaluations at steps 5,000, 10,000, and 15,000. Select a checkpoint using validation data only:

1. every lexical, pitch, voicing, clipping, and artifact gate passes;
2. highest worst-case lyric similarity;
3. lowest worst-case pitch error;
4. earliest checkpoint.

The Korean held-out set covers ordinary, rapid, large interval, sustained note, repeated note, high register, and phrase boundary cases. Each case is rendered with seeds 7, 21, and 42.

Mandatory Korean foundation gates are:

- normalized free-Whisper lyric similarity at least 0.90 for every phrase and seed;
- no repeated or omitted lyric span;
- mean RMVPE pitch MAE at most 20 cents and every row at most 35 cents;
- pitch p90 absolute error at most 60 cents;
- gross pitch error rate at most 3%;
- voicing accuracy at least 0.90 for every row;
- zero clipping samples;
- no material HF-spike, sample-jump, waveform-discontinuity, or multi-resolution-STFT regression against its qualified source reference;
- all WAVs have deterministic names, paths, and SHA-256 values.

Failure produces `foundation_ko_gate_reject` and stops before GYU identity training.

## Phase 3: multilingual foundation

After the Korean gate passes, add source-qualified Japanese soprano and English alto rows. Reuse the same qualification function and thresholds. The previous Japanese soprano checkpoint is evidence, not a hidden shortcut; the new multilingual checkpoint must pass its own frozen matrix.

Use language-tagged phonemes and separate speaker IDs. Train one shared acoustic foundation with explicit language and speaker conditioning. Freeze train/validation/test splits by complete song and singer before optimization. Fixed multilingual evaluation contains at least five held-out phrases per language and seeds 7, 21, and 42.

The multilingual checkpoint must preserve the Korean gates and meet the same lexical, pitch, voicing, clipping, and artifact rules for EN and JA. A language failing one mandatory row rejects the checkpoint. No easier replacement phrase is allowed after results are visible.

## Phase 4: bounded GYU identity adaptation

The multilingual foundation supplies the linguistic, duration, score, pitch, decoder, and vocoder behavior. These parameters are frozen initially:

- phoneme and language encoders;
- duration and variance predictors;
- pitch and voicing predictors;
- acoustic decoder and mel projection;
- vocoder;
- non-GYU speaker rows.

Train only:

- one new GYU speaker embedding;
- one zero-initialized, bounded speaker-conditioning adapter at the existing speaker-conditioning point;
- normalization scale/bias only if explicitly listed in the frozen manifest before training.

The initialized adapter must reproduce the unadapted foundation within documented floating-point tolerance. A parameter and gradient audit must prove that frozen parameters do not receive gradients.

Use real GYU data with explicit trust labels:

- 24 independently verified-score phrases: weight 1.0;
- high-confidence inferred-score phrases: weight 0.50 to 0.75, fixed before training;
- lower-confidence inferred rows: excluded;
- external GTSinger rows: foundation replay only, never labeled GYU;
- synthetic or teacher speech: excluded from GYU identity targets.

The preservation target is the aligned unadapted foundation render. Losses include WavLM and ECAPA identity, phoneme/content preservation, duration/timing preservation, pitch and voicing preservation, waveform preservation, FFT 256/1024/4096 multi-resolution STFT preservation, adapter-output regularization, and parameter-update regularization. Log each component separately.

Training is bounded by a frozen maximum step count and fixed validation intervals. Checkpoint selection is lexicographic: all preservation gates pass, then both speaker metrics improve consistently, then the smallest update, then the earliest checkpoint. Held-out data cannot select a checkpoint or change a weight.

## Phase 5: final machine evaluation

Compare the unadapted multilingual foundation, initialized GYU adapter, and selected trained candidate under identical scores and seeds.

Evaluate:

- all frozen KO stress phrases x seeds 7/21/42;
- five or more held-out EN phrases x seeds 7/21/42;
- five or more held-out JA phrases x seeds 7/21/42;
- rapid, sustained, large-interval, high-note, repeated-note, and phrase-boundary cases;
- a long-form copyrighted-free or project-authored score.

For every WAV record the expected lyric, free-Whisper transcript, lyric similarity, repetition/omission evidence, RMVPE pitch MAE, pitch p90, gross pitch error, voicing accuracy, clipping, HF spike, sample jump p99.9, waveform discontinuity, FFT 256/1024/4096 comparisons, WavLM and ECAPA similarity against every fixed GYU reference, runtime, memory, path, and SHA-256.

The GYU candidate must improve held-out mean WavLM and ECAPA similarity by at least 0.01 each over the matching unadapted foundation. At least 80% of phrase-seed rows must improve both metrics, no fixed reference may show an aggregate decline, and no individual identity regression may exceed 0.01. Identity gain cannot compensate for any lexical, pitch, voicing, clipping, or artifact failure.

A machine pass is labeled only `diagnostic_candidate_human_pending`.

## Phase 6: human listening and OpenUtau package

Create a compact blind A/B set only after every machine gate passes. It includes KO neutral, rapid, energetic, sustained, high, large interval, EN, JA, repeated notes, phrase boundary, and long-form output. The user must explicitly approve intelligibility, note timing, pitch stability, continuity, absence of metallic artifacts, and GYU identity.

Only an explicitly approved candidate may be exported using the existing DiffSinger/OpenUtau package path. The package must include the acoustic model, duration model, dictionaries, language maps, phonemizer configuration, attribution, CC BY-NC-SA 4.0 notice, model/data revisions, checksums, and a release manifest. It must not include source recordings, external datasets, caches, or training checkpoints.

Validate the package in a clean local environment by installing it into OpenUtau and rendering fixed KO/EN/JA USTX fixtures. The clean renders must match the approved backend within documented ONNX tolerance and pass the same core lyric, pitch, voicing, clipping, and artifact gates.

The local package is not a public release. Public distribution requires a separate license decision and the required attribution/share-alike terms.

## Failure handling and truthful status

Stop at the first mandatory gate that invalidates later work:

- insufficient qualified source: `foundation_source_gate_reject`;
- failed Korean foundation: `foundation_ko_gate_reject`;
- failed multilingual foundation: `multilingual_foundation_reject`;
- identity feasibility failure: `identity_training_feasibility_failure`;
- preservation or identity failure: `gyu_identity_candidate_reject`;
- machine pass awaiting listening: `diagnostic_candidate_human_pending`;
- human rejection: `human_listening_reject`.

Rejected evidence is preserved but never referenced by the production renderer or package. A failed experiment is not goal completion. The goal is complete only after the user passes listening and the clean OpenUtau installation/render verification succeeds.

## Repository and evidence policy

Commit only implementation, tests, compact frozen manifests, compact metric summaries, configuration, commands, and reports. Keep source recordings, GTSinger audio, rendered WAVs, checkpoints, caches, and large plots under ignored local paths. Original files under `data/source/` remain unchanged.

After runtime changes, run the full relevant tests, `python scripts/validate_dataset.py`, package smoke tests, and `git diff --check`. No main merge, push, PR, package publication, or production pointer change is part of the experiment without later explicit authorization.
