# Post-RC4 audio-quality candidate 4

Status: objective candidate; human listening pending. This is not a tag or release.

## Changes

- canonical 50 Hz phoneme/note/voicing/user-pitch timeline;
- F0=0 for inferred/explicit unvoiced consonants and silence;
- OpenUtau timed phones accepted by the timeline; inferred subdivisions are explicitly labeled;
- OpenUtau pitch curve remains authoritative; only explicit slurs receive a 60 ms transition when no pitch curve exists;
- CTC timing correction is applied inside SoulX content hidden state, never by waveform pitch shifting or phase vocoding;
- decoder settings selected per measured stress regime: KO/EN 32 steps CFG1.5, rapid 64 steps CFG2.0, large-interval 32 steps CFG2.0;
- rapid uses CTC phoneme-hold hidden timing; EN uses 0.25 timing correction;
- safety peak gain only when needed; no denoising or dereverberation was applied.

## Objective result

`artifacts/reports/rc5_stress_candidate4/evaluation.json`:

- 9/9 at 48 kHz, zero clipping;
- all lyric coverage ≥0.8;
- core four versus RC4: HF-spike ratio -15.3%, spectral flux -7.0%, sample jump -24.4%;
- pitch MAE +1.81 cents;
- ASR similarity -0.025.

The aggregate HF-energy p95 is slightly higher on the core four (+0.00072), concentrated in rapid KO. The corrected Korean obstruent mask initially collapsed the 64-step large-interval decode; a controlled same-input sweep selected 32 steps, CFG 2.0, seed 21 with exact “높이 날아” ASR and 17.78-cent pitch MAE. Human listening must decide whether the large reductions in spikes/discontinuity correspond to a clearly less metallic result. Automated metrics cannot pass this gate.

## Listening

- nine-case set: `artifacts/reports/rc5_stress_candidate4/listening/`
- matched RC4 before/after windows: `artifacts/reports/rc5_stress_candidate4/before_after/`

Remaining risks: rapid KO retains elevated HF energy; EN ASR says “Sink” for “Sing”; sustained “아” is transcribed as “아멘” although lyric coverage is complete. Do not publish final v1.0.0. If human listening fails, continue from candidate4 rather than tagging it.

## Verification

- full tests: 37 passed;
- dataset validation: 132 recordings, 0 corrupt;
- clean wheel install: pass in an isolated `/tmp` virtual environment;
- installed KO/EN/JA frontend and internal candidate renderer import: pass;
- reference audit: retain original lossless-derived PCM; no generic denoise, dereverb, or UVR.

This verifies implementation integrity only. It does not replace the pending human listening gate, and no RC5 or final-release package has been created.
