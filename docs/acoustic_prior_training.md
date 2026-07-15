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
