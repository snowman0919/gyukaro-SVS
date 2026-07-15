# Acoustic prior training

The compatible public prior uses LibriTTS-R and VocalSet, both CC BY 4.0. 127 of 144 measured candidates passed quality gates: 60 LibriTTS-R and 67 VocalSet. Splits are speaker-disjoint. Original archives and source audio are not packaged.

Real SoulX reconstruction created 78 degradation pairs (24 LibriTTS-R, 30 VocalSet, 24 real GYU), split 44/17/17. No random-noise corruption or synthetic codec degradation was used.

The universal residual backbone has 51,537 parameters, with 41,601 trained for 800 steps. Validation loss was 1.948519. Its bounded residual is activity-gated, so silent regions cannot receive an invented noise floor.

## Korean score-native probe

Zeroth-Korean contributes a separate 400-row, four-speaker, CC BY 4.0 speech subset. Its MMS CTC labels are inferred and it is never described as singing supervision. Full acoustic adaptation without singing replay caused catastrophic forgetting; VocalSet replay bounded the damage but did not restore Korean lyrics. The best replay checkpoint reached 0.191 stress-set ASR similarity and was rejected.

Real-GYU CTC segmentation recovered 730 source-derived acoustic segments without modifying `data/source/` and without using the 24 independent-score evaluation rows. A bounded GYU-only adaptation reached 0.311 ASR similarity, while rapid output still collapsed to `아`. It was rejected before runtime integration. The production backend and RC6 behavior are unchanged.

GTSinger and CSD would provide lexical score-aligned singing, but their verified licenses impose non-commercial/ShareAlike restrictions. They were recorded as excluded in the license registry and were not downloaded. No compatible lexical singing prior has therefore passed the production-data gate.

The public-domain `Row Your Boat` excerpts already present in VocalSet provide a compatible lexical singing subset: 76 speaker-disjoint phrase rows, 8.365 minutes, with inferred MMS CTC timing. Training them jointly with all real-GYU segments did not improve Korean stress lyrics over the GYU-only checkpoint (both peaked at 0.311 ASR similarity); later checkpoints regressed voicing and pitch. The branch is retained as a reproducible negative result, not as an RC model.
