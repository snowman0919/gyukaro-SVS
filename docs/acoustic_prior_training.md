# Acoustic prior training

The compatible public prior uses LibriTTS-R and VocalSet, both CC BY 4.0. 127 of 144 measured candidates passed quality gates: 60 LibriTTS-R and 67 VocalSet. Splits are speaker-disjoint. Original archives and source audio are not packaged.

Real SoulX reconstruction created 78 degradation pairs (24 LibriTTS-R, 30 VocalSet, 24 real GYU), split 44/17/17. No random-noise corruption or synthetic codec degradation was used.

The universal residual backbone has 51,537 parameters, with 41,601 trained for 800 steps. Validation loss was 1.948519. Its bounded residual is activity-gated, so silent regions cannot receive an invented noise floor.
