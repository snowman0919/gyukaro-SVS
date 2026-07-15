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
