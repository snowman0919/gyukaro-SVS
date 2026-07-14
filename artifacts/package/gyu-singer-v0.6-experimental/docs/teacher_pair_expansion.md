# v0.6 teacher pair expansion

`scripts/build_teacher_internal_pairs.py` groups existing quality-gated benchmark IDs; it does not duplicate semantic examples. The Fish/MOSS intersection is 191 IDs: KO 99, EN 77, JA 15. Higgs is present as waveform-level evidence for 180 IDs but no stable Higgs hidden representation was exposed. Splits are benchmark-ID disjoint (train 165, validation 9, test 17). `scripts/extract_teacher_representations.py` extracted 382 real hidden vectors (191 per teacher).

Fish uses `Fish-S2-Pro-DAC.encoder_hidden`; MOSS uses `MOSS-Audio-Tokenizer-Nano.encoder_hidden_states`, mean-pooled over time. Trust and cross-teacher agreement remain metadata weights; speech-teacher rows are not real GYU singing labels.
