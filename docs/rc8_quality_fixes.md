# RC8 local quality candidate

Status: objective non-regression passed; human listening pending. RC8 is not accepted and RC9 has not started.

## Scope and preserved baseline

RC7 remains frozen at `ae8944070f3dc38e310b33f29d95f4bcd3c81def`; its WAVs and checkpoint hashes are recorded in `docs/rc7_baseline.md`. RC8 writes only new artifacts and retains phrase-level SoulX decoding, 48 kHz PCM-24 output, the RC7 base spectral correction at strength 0.5, and the protected Rapid KO 64-step/CFG 2.0 policy. It uses no per-note TTS, waveform pitch shifting, or phase-vocoder note control.

## Diagnosis and bounded changes

| Defect | Isolated source | Selected RC8 change | Evidence |
|---|---|---|---|
| Sustained noise | Stable vowels benefited from stronger spectral correction; transient regions did not justify it | Extra 0.5 correction only inside Korean voiced runs that are stable for at least 300 ms, away from note/voicing boundaries, and only when a note lasts at least 2.5 s | `artifacts/reports/rc8_sustained_set/evaluation.json` |
| English transitions | ARPAbet `AY` was absent from the vowel set, forcing F0=0 through words such as *light* and *sky* | Classify `AY` as voiced and keep EN timing warp at 0.25 | `artifacts/reports/rc8_frontend_candidate/evaluation.json` |
| Korean overconnection | Generated-content/score phone-center mismatch was median 375 ms; warp 0.1 improved inferred closures but changed a lyric in the actual backend | Use only 0.05 warp, only on neutral contiguous non-interval KO; Rapid remains on its existing hold path | `artifacts/reports/rc8_ko_timing_sweep/evaluation.json` |
| Japanese attenuation | Note-by-note chunks missed phrase-only lexicon keys and became inferred unvoiced `ja_unknown_*` phones | Add the exact score chunks to the Japanese lexicon; do not amplify legitimate devoicing | `artifacts/reports/rc8_frontend_candidate/evaluation.json` |
| Large interval | The wrong apparent trajectory was already present before the spectral refiner: the 32-step decode tracked about 779 Hz in the first failure region | Use SoulX 50 steps/CFG 2.0 and do not apply the stronger stationary gate | `artifacts/reports/rc8_interval_candidate/evaluation.json` |

The first full RC8 attempt was rejected because over-broad timing/spectral gates reduced mean ASR to 0.732. The second attempt restored Rapid and Interval content but normal KO still changed *작은* to *큰*. The selected 0.05 KO warp preserves the full expected transcript. These rejected candidates remain under `artifacts/reports/rc8_backend_candidate/` and `artifacts/reports/rc8_backend_candidate2/` as negative evidence.

## Multi-resolution definitions

The repository had no prior SR/MR definition for this gate. RC8 defines three explicit 48 kHz analysis scales:

- short: FFT 256, hop 64, 5.33 ms window for attacks and consonants;
- medium: FFT 1024, hop 256, 21.33 ms window for phoneme/formant behavior;
- long: FFT 4096, hop 1024, 85.33 ms window for sustained harmonic stability.

On six sustained cases, strength 0.5 to the stationary-gated equivalent of strength 1.0 changed HNR proxy 22.63 to 23.41 dB, short/medium/long instability 0.02046/0.01316/0.03392 to 0.01866/0.01183/0.03095, and sample jump 0.03030 to 0.02231. Pitch MAE stayed 7.93 cents; voicing changed 0.9733 to 0.9707. This small voicing decrease is why human listening remains mandatory.

In the Large Interval failure region, RC7 to the selected 50-step candidate changed pitch MAE 15.42 to 11.41 cents, voiced accuracy 0.6625 to 0.6958, long-window instability 0.00856 to 0.00575, and long-window noise modulation 16.66 to 12.65 dB. The primary pYIN track changed from the erroneous 779 Hz trajectory to 387 Hz. A second estimator still disagrees by roughly an octave, so the mechanical defect cannot be declared fixed without listening.

## Nine-file objective gate

RC7 and RC8 were evaluated against target F0 rebuilt with the current frontend.

| Metric | RC7 | RC8 candidate |
|---|---:|---:|
| ASR lyric similarity | 0.924211 | 0.930667 |
| ASR lyric coverage | 0.952189 | 0.952189 |
| Pitch MAE, cents | 9.097778 | 7.713333 |
| Voicing accuracy | 0.868011 | 0.873289 |
| Spectral flux p95 | 0.229099 | 0.226921 |
| Sample jump p99.9 | 0.073507 | 0.072940 |
| HF spike p99/median | 344.181322 | 362.049967 |
| WavLM-to-GYU | 0.617891 | 0.619059 |
| ECAPA-to-GYU | 0.118516 | 0.122516 |

The candidate passes ASR, pitch, voicing, clipping, and identity non-regression. It does not achieve a global 5% HF-spike reduction; English and Japanese local fixes increase that aggregate. Therefore the report status is `objective_nonregression_human_pending`, not “materially improved.” The listening decision is authoritative for the remaining perceptual defects.

Human review passed eight cases and rejected Large Interval. A follow-up score-domain sweep found an 80 ms large-jump onset transition that keeps exact ASR while changing the failure-region pYIN/YIN disagreement from about 1203 to 4 cents. Relative to the failed RC8 interval file, pitch MAE improves from 11.41 to 10.55 cents, voicing from 0.6958 to 0.7292, and HF spike p99/median from 92.55 to 50.71. This candidate remains listening-pending and is not yet part of the backend.

Validation: 49 tests passed; dataset validation passed for 132 recordings (`106..237`, mono 48 kHz, corrupt 0). The nine-file actual-backend render is the RC8 runtime smoke. No RC8 package is created before human acceptance; clean package validation belongs to RC9.
