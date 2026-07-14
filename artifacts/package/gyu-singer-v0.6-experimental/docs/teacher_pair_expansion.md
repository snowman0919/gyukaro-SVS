# v0.6 teacher pair expansion

`scripts/generate_v06_teacher_expansion.py` generated 70 unique semantic text/reference prompts per mandatory teacher: 30 KO, 20 EN, and 20 JA. Fish S2 Pro and MOSS Local Transformer generated real WAVs; generated audio stays ignored under `data/teacher/`.

`scripts/evaluate.py` then gated 140 new teacher outputs. 127 passed. Combining them with the existing quality-gated corpus produced 249 Fish/MOSS internal pairs from 102 unique `(language, text, reference_ids)` semantic groups: KO 128, EN 92, JA 29; train/validation/test 191/32/26. Semantic groups are assigned atomically, so no text/reference group crosses a split.

Fish representation is `Fish-S2-Pro-DAC.encoder_hidden`; MOSS is `MOSS-Audio-Tokenizer-Nano.encoder_hidden_states`. Both are mean-pooled real internal neural states, not waveform STFT summaries. `scripts/extract_teacher_representations.py` saved 498 vectors (249 per teacher). Higgs remains waveform-level auxiliary evidence for 180 rows because no stable official hidden-state extraction was available.

The original 44-group pilot is retained only as historical v0.5 evidence. The v0.6 identity checkpoint is trained/evaluated on the 102-group corpus.
