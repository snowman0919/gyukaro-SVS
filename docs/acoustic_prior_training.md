# Acoustic prior training

The compatible public prior uses LibriTTS-R and VocalSet, both CC BY 4.0. 127 of 144 measured candidates passed quality gates: 60 LibriTTS-R and 67 VocalSet. Splits are speaker-disjoint. Original archives and source audio are not packaged.

Real SoulX reconstruction created 78 degradation pairs (24 LibriTTS-R, 30 VocalSet, 24 real GYU), split 44/17/17. No random-noise corruption or synthetic codec degradation was used.

The universal residual backbone has 51,537 parameters, with 41,601 trained for 800 steps. Validation loss was 1.948519. Its bounded residual is activity-gated, so silent regions cannot receive an invented noise floor.

## Korean score-native probe

Zeroth-Korean contributes a separate 400-row, four-speaker, CC BY 4.0 speech subset. Its MMS CTC labels are inferred and it is never described as singing supervision. Full acoustic adaptation without singing replay caused catastrophic forgetting; VocalSet replay bounded the damage but did not restore Korean lyrics. The best replay checkpoint reached 0.191 stress-set ASR similarity and was rejected.

Real-GYU CTC segmentation recovered 730 source-derived acoustic segments without modifying `data/source/` and without using the 24 independent-score evaluation rows. A bounded GYU-only adaptation reached 0.311 ASR similarity, while rapid output still collapsed to `아`. It was rejected before runtime integration. The production backend and RC6 behavior are unchanged.

Scaling the balanced Zeroth prior to 1,400 rows (3.544 hours) made lexical transfer worse at the matched 300-step gate: ASR similarity fell from 0.191 to 0.077 and HF spikes exceeded RC6. The run was stopped at its saved checkpoint; no longer continuation is justified. This is speech supervision only and cannot replace compatible Korean lexical singing supervision.

GTSinger and CSD would provide lexical score-aligned singing, but their verified licenses impose non-commercial/ShareAlike restrictions. They were recorded as excluded in the license registry and were not downloaded. No compatible lexical singing prior has therefore passed the production-data gate.

The public-domain `Row Your Boat` excerpts already present in VocalSet provide a compatible lexical singing subset: 76 speaker-disjoint phrase rows, 8.365 minutes, with inferred MMS CTC timing. Training them jointly with all real-GYU segments did not improve Korean stress lyrics over the GYU-only checkpoint (both peaked at 0.311 ASR similarity); later checkpoints regressed voicing and pitch. The branch is retained as a reproducible negative result, not as an RC model.

Two additional real-GYU controls also failed. A low-rate GYU stage after the speech/singing replay prior collapsed to repeated syllables. Filtering out exercise-heavy recordings reduced the high-frequency spike proxy, but its short isolated segments still emitted vowels instead of Korean lyrics. Rebuilding those labels into 81 multi-syllable phrase chunks at 800 ms gaps did not fix lexical generation: the best checkpoint reached 0.311 ASR similarity and 0.695 voicing accuracy. These inferred-label probes remain reproducible negative evidence; none changes the runtime or RC6.

## Expanded real-degradation refiner v2

The v2 universal stage used 52 LibriTTS-R degradation pairs (27 train, 14 validation, 11 test) rather than 18 total baseline rows. Its validation loss improved from 1.9485 to 1.8386. The singing stage used 221 VocalSet pairs (130 train, 47 validation, 44 test) plus 27 speech replay rows; only the 4,968-parameter singing adapter was trainable.

On held-out VocalSet speakers, singing-v2 at 25% improved log-spectral L1 from 0.6274 to 0.5873, high-band L1 from 0.5322 to 0.4683, and the HF-spike proxy from 1062 to 334. On the actual nine-file RC stress set, however, 25% reduced ASR and voicing. Reducing strength to 15% preserved ASR but only reduced HF spikes 4.6% while voicing still regressed. This is not a material listening candidate. Both v2 checkpoints remain experimental and the production runtime is unchanged.

## Aligned spectral refiner

The waveform TCN's short receptive field was replaced in an isolated probe by a 180,257-parameter identity-initialized STFT-mask U-Net. Its six bottleneck blocks expose separate 6,576-parameter singing/GYU adapters. It changes magnitude conditioning while retaining source phase, pitch timeline, and waveform length. No random corruption or generic denoising data is used.

Global pair alignment was measured before training. LibriTTS-R p95 absolute lag is 4.5 ms; VocalSet p95 is 50 ms, with one 420 ms outlier. Training applies deterministic 10 ms RMS-envelope correlation within ±500 ms. The universal stage used 27/14/11 speaker-disjoint LibriTTS-R train/validation/test rows and reached validation loss 1.664884, versus 1.8386 for waveform-v2.

On the nine human-failed RC6 files, the selected 50% spectral-singing strength preserves mean ASR exactly at 0.924211, improves pitch MAE 8.4689→8.2411 cents, voicing 0.8720→0.8727, HF spikes 458.56→344.18, sample jumps 0.09360→0.07351, WavLM-to-GYU 0.59999→0.61789, and ECAPA-to-GYU 0.11205→0.11852. This is objective evidence only; the model remains outside production until human review.
