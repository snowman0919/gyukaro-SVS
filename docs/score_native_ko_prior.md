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
