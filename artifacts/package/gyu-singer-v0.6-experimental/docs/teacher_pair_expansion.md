# v0.6 teacher pair expansion

`scripts/build_teacher_internal_pairs.py` groups existing quality-gated benchmark IDs. The Fish/MOSS intersection is 191 style-conditioned rows: KO 99, EN 77, JA 15, but only 44 unique `(language, text, reference_ids)` groups. This does not meet the preferred 100+ unique semantic-pair target and must not be presented as a large unique-pair corpus. Higgs is present as waveform-level evidence for 180 IDs but no stable Higgs hidden representation was exposed. Text/reference groups are assigned atomically to train/validation/test (149/22/20 rows), preventing semantic leakage. `scripts/extract_teacher_representations.py` extracted 382 real hidden vectors (191 per teacher).

Fish uses `Fish-S2-Pro-DAC.encoder_hidden`; MOSS uses `MOSS-Audio-Tokenizer-Nano.encoder_hidden_states`, mean-pooled over time. Trust and cross-teacher agreement remain metadata weights; speech-teacher rows are not real GYU singing labels.
