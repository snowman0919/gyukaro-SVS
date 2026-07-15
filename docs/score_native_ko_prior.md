# Score-native Korean phoneme prior probe

Status: rejected engineering probe; not an RC and not a release path.

The 4k direct DiffSinger pilot recovered score pitch but emitted mostly vowel-like content. Its training corpus contains 1,997 VocalSet rows with only non-lexical vowels and 43 real GYU training phrases, so Korean consonant acoustics are severely underrepresented.

Sixty existing Korean Fish/MOSS teacher utterances were force-aligned with MMS CTC. All labels are explicitly `inferred_mms_ctc_low_trust_teacher_speech`; these rows are phoneme priors, not real GYU singing. IDs 026–030 are held out for both teachers. The isolated binary contains 93 training rows and 19 validation rows. Six previously unseen Korean phones were added through a token-name-preserving checkpoint remap; shared embedding error is zero.

| Probe | Pitch MAE | Voicing accuracy | ASR similarity | Decision |
|---|---:|---:|---:|---|
| Direct pilot 4k | 31.27 cents | 0.6691 | 0.2250 | reject |
| Text path only, best ASR | 5.76 cents | 0.7497 | 0.2000 | reject |
| Acoustic path, 300 steps | 9.57 cents | 0.6825 | 0.1361 | reject |

The text-only adaptation improves pitch stability but does not restore lyrics. Allowing the acoustic decoder to learn the same tiny speech set increases high-frequency spikes and still does not restore lyrics. More training on these 60 synthesized rows is therefore stopped.

The next bounded experiment is a speaker-balanced subset of the CC BY 4.0 Zeroth-Korean corpus, used only for Korean phoneme/acoustic pretraining, followed by real-GYU singing adaptation. Full-corpus download and blind end-to-end retraining remain out of scope. Objective metrics may reject a probe; human listening is still required to accept any future candidate.

## Zeroth and contiguous-GYU follow-up

Status: rejected; none of these checkpoints is an RC candidate.

The bounded Zeroth subset contains 400 unique utterances from four speakers (320 train, 80 validation, 1.025 hours). MMS CTC timings are inferred and Zeroth remains a speech-only phoneme/acoustic prior. Acoustic adaptation without singing replay catastrophically forgot singing. Restoring all 1,955 VocalSet training rows prevented the worst high-frequency collapse, but the best replay result still reached only 0.191 ASR lyric similarity.

The real-GYU phrase labels exposed a separate defect: 62.38% of their duration was `SP`, including 316 gaps of at least 200 ms. Deterministic source-preserving segmentation reduced this to 33.76% over 230 segments. Extending CTC alignment to every usable recording recovered 89 source rows and 730 segments (8.69 minutes, 25.33% `SP`); all 24 independent-score rows remained excluded. Seven unseen Korean tokens were initialized from their same-category embedding mean, while every shared embedding was preserved exactly.

| Probe | Pitch MAE | Voicing | HF spike | ASR similarity | Decision |
|---|---:|---:|---:|---:|---|
| RC6 (human failed) | 20.64 cents | 0.802 | 703.81 | 0.967 | baseline only |
| Zeroth + VocalSet replay 300 | 7.04 cents | 0.674 | 248.61 | 0.191 | reject |
| 230 contiguous GYU segments 200 | 9.12 cents | 0.714 | 103.15 | 0.291 | reject |
| 730 all-recording segments 400 | 10.47 cents | 0.651 | 304.59 | 0.300 | reject; discontinuity regression |
| GYU-only adaptation 100 | 6.27 cents | 0.730 | 181.93 | 0.311 | reject |

Rapid Korean still transcribes as `아`, and large-interval Korean loses consonants. More steps consistently reduce voicing and lyric retention. The direct path therefore lacks a compatible score-native lexical singing prior; speech data, non-lexical VocalSet vowels, and additional optimization do not substitute for one. GTSinger and CSD were not downloaded because their non-commercial/ShareAlike terms are incompatible with the intended redistributable production checkpoint.

## VocalSet lexical-singing follow-up

Status: valid generic prior data, rejected Korean transfer probe.

The existing CC BY 4.0 VocalSet archive also contains 38 `Row Your Boat` recordings with real public-domain lyrics. Straight and vibrato performances were MMS-CTC aligned, split into 76 phrase rows (20 singers, 8.365 minutes), and separated by singer into 62 training and 14 validation rows. Timings remain explicitly inferred. The pilot replayed all 730 GYU segments, kept score-control layers frozen, and used a `1e-5` learning rate.

| Probe | Pitch MAE | Voicing | HF spike | ASR similarity | Decision |
|---|---:|---:|---:|---:|---|
| GYU-only adaptation 100 | 6.27 cents | 0.730 | 181.93 | 0.311 | prior best; reject |
| VocalSet lexical 100 | 9.00 cents | 0.719 | 193.20 | 0.311 | no transfer; reject |
| VocalSet lexical 200 | 60.84 cents | 0.486 | 186.68 | 0.111 | collapse; reject |
| VocalSet lexical 300 | 50.99 cents | 0.451 | 132.32 | 0.000 | collapse; reject |

The lexical recordings are technically usable, but English consonant supervision does not supply a Korean lexical singing prior. This closes the generic-English-transfer branch without changing RC6 or requesting human review.

## Real-GYU adaptation and phrase-chunk follow-up

Status: rejected; neither checkpoint family is an RC candidate.

The 730-segment real-GYU corpus over-samples recording exercises: 475 rows come from blocks A/B, while the strict lexical subset retains 243 rows (4.299 minutes). A low-rate second-stage adaptation after the Zeroth replay prior and a GYU-only lexical adaptation both preserved score pitch but emitted repeated vowels or syllables. The lexical data also exposed a segmentation defect: 131 of 243 rows were shorter than one second because a 250 ms gap threshold split coarticulation context.

Phrase rebuilding at 800 ms gaps produced 81 inferred-timing chunks from 37 source recordings (4.742 minutes, 3.454-second median). All 24 independent-score rows remained excluded and original recordings were unchanged.

| Probe | Pitch MAE | Voicing | HF spike | ASR similarity | Decision |
|---|---:|---:|---:|---:|---|
| Zeroth then all-GYU, 100 | 6.85 cents | 0.692 | 209.52 | 0.000 | repetition collapse |
| Lexical GYU, 100 | 6.78 cents | 0.693 | 126.76 | 0.222 | vowel collapse |
| Phrase chunks, 100 | 8.26 cents | 0.709 | 119.01 | 0.222 | vowel collapse |
| Phrase chunks, 200 | 7.82 cents | 0.695 | 132.96 | 0.311 | objective reject |
| Phrase chunks, 300 | 8.70 cents | 0.629 | 144.07 | 0.111 | regression |

Longer chunks improve the high-frequency proxy but do not recover Korean lyrics: rapid output remains `아`, while the 300-step interval case collapses to a sustained `으`. The root defect is therefore not just boundary fragmentation. This branch needs a compatible pretrained Korean lexical singing model; more optimization on the current small inferred corpus is stopped.

## Official Korean MLP Singer and GYU conversion follow-up

Status: the official score-native source passes the bounded Korean source probe; every tested GYU conversion is rejected.

The MIT-licensed official `neosapience/mlp-singer` checkpoint (`7f4621c`) reproduces its Korean sample and directly accepts phonemes, durations, notes, and F0. Increasing onset-consonant allocation to 70 ms (`c6`) gave 0.900 aggregate ASR similarity on the independent rapid and large-interval stress scores, 18.70-cent pitch MAE, and far fewer high-frequency spikes than RC6. This proves that score-native phoneme timing addresses the usability defect. It is still a generic CSD singer, not GYU.

Full and projection-only adaptation on 43 inferred-score GYU training phrases forgot lyrics. A 10% weight blend improved WavLM similarity by less than 0.01 and reduced ECAPA similarity, so it was not accepted. Converting the unadapted source through SoulX reduced ASR similarity from 0.900 to 0.383.

As a final bounded identity-conversion test, RVC v2 48k was trained on 518 segments (19.52 minutes) derived from 66 real GYU recordings. No independent verified-score row was used and no source recording was denoised, dereverberated, or modified. The continuation loaded the epoch-5 generator but reset the discriminator and optimizer, so it is labeled `e5_plus15`, not continuous e20.

| Probe | ASR similarity | Pitch MAE | Voicing | HF spike | WavLM GYU | ECAPA GYU | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Score-native source c6 | 0.900 | 18.70 | 0.878 | 258.13 | 0.701 | 0.133 | generic source only |
| RVC e5 | 0.259 | 19.93 | 0.958 | 661.13 | 0.751 | 0.200 | reject: lyrics collapse |
| RVC e5 + 15 | 0.444 | 20.41 | 0.953 | 1161.82 | 0.774 | 0.186 | reject: lyrics and HF regress |

The measurable identity gain does not compensate for semantic destruction or metallic-artifact growth. SoulX and RVC are therefore both rejected as post-source GYU conversion paths. More optimization on either conversion is stopped; the next viable architecture must preserve the score-native acoustic source and condition identity inside that model or use a compatible pretrained Korean lexical SVS foundation.

### Bounded latent identity adapters

Three adapters were inserted between the frozen MLP Mixer decoder and frozen mel projection: a 576-parameter FiLM trained on GYU mel L1, the same FiLM trained by frozen WavLM+ECAPA agreement, and a 9.5k-parameter zero-initialized low-rank residual trained by the same dual-speaker objective. All retain the score, phoneme, mixer, projection, and official HiFi-GAN weights.

| Probe | ASR similarity | HF spike | WavLM GYU | ECAPA GYU | Decision |
|---|---:|---:|---:|---:|---|
| Score-native source c6 | 0.900 | 258.13 | 0.701 | 0.133 | generic source only |
| mel-FiLM 100 | 0.842 | 293.92 | 0.672 | 0.140 | reject: no cross-encoder gain |
| speaker-FiLM 25 | 0.775 | 236.52 | 0.729 | 0.130 | reject: ECAPA and lyrics regress |
| speaker-residual 25 | 0.775 | 234.63 | 0.729 | 0.131 | reject: ECAPA and lyrics regress |
| speaker-residual 100 | 0.475 | 248.86 | 0.718 | 0.127 | reject: rapid lyric collapse |

The adapter family is therefore closed. It cannot reach GYU identity without sacrificing the score-native source's semantic advantage on this small inferred-score corpus.

The source model itself also cannot be promoted to production. Although its repository distributes code and weights under MIT, its underlying Children's Song Dataset archive is marked `CC BY-NC-SA 4.0` by the actual Zenodo record (`4916302`). The paper footer says CC BY 4.0, but the downloadable artifact's metadata is the governing conservative evidence for this project. CSD replay and derived production weights remain excluded.

## Scaled speech-prior and official SoulX direct controls

Status: rejected; scaling speech and projecting Korean onto SoulX's non-Korean phones do not solve lexical singing.

The same four-speaker Zeroth selection was expanded from 400 to 1,400 unique utterances (3.544 hours; 1,260 train and 140 validation rows). At the saved 300-step replay checkpoint, ASR lyric similarity fell from 0.191 to 0.077 and the HF-spike ratio rose from 248.61 to 881.09. Pitch improved from 7.04 to 5.76 cents, but output text became unrelated speech. No 600/1,200-step continuation was authorized because scale produced no lexical-transfer signal.

The official SoulX `model.pt` score path was then tested directly with note pitch, duration, and phonemes. Korean phones were deterministically approximated with the model's English ARPAbet vocabulary, so this is explicitly not native Korean supervision. A verified real-GYU prompt improved the score-control probe from 0.000 to 0.400 ASR similarity, but large-interval content collapsed. Replacing note-pitch control with canonical 50 Hz language-aware F0 improved pitch from 27.50 to 4.16 cents and voicing from 0.864 to 0.974, while ASR fell to 0.100 and HF spikes remained worse than RC6 (842.55 versus 703.81).

These controls isolate two facts: canonical F0 fixes pitch/voicing conditioning, but neither a larger Korean speech prior nor a non-native phone projection preserves Korean lexical content through the singing decoder. More SoulX step/CFG sweeps on this projection are stopped.

| Probe | ASR similarity | Pitch MAE | Voicing | HF spike | Decision |
|---|---:|---:|---:|---:|---|
| Zeroth 1.025 h replay 300 | 0.191 | 7.04 | 0.674 | 248.61 | reject |
| Zeroth 3.544 h replay 300 | 0.077 | 5.76 | 0.622 | 881.09 | reject |
| SoulX score + verified prompt | 0.400 | 27.50 | 0.864 | 1638.16 | reject |
| SoulX melody + canonical voicing | 0.100 | 4.16 | 0.974 | 842.55 | reject |
