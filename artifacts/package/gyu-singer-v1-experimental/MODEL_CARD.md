## Scope

Input: Korean, English, or Japanese lyric-note JSON. Model: TriSinger conditional-flow acoustic-latent generator (3.0 MB checkpoint) decoded by frozen Apache-2.0 MOSS audio tokenizer. Training: 76 real GYU segments; 60 train/5 validation/5 test, real anchors only for acoustic target; 633 teacher rows representation-only at trust 0.05-0.20. Score for real anchors was inferred from speech duration, not ground-truth singing notation.

This is an experimental personalized SVS runtime, not a production-quality multilingual singer.
