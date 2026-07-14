# v0.5 training stages

1. Score reconstruction: 76 inferred RMVPE/script-constrained rows; 76 accepted after confidence gate.
2. Singing alignment: 76 MMS CTC rows with Korean onset/nucleus/coda prior.
3. Teacher representation: Fish S2 Pro DAC + MOSS tokenizer internal hidden states; 4 paired rows, shared 32-D projection.
4. Real prosody: 76 real GYU targets, 65 train / 6 validation / 5 test; nominal score only as condition.
5. Acoustic adapter: 76 real GYU rows plus 32 low-trust style rows; final spectral adapter checkpoint.

Training reports record optimizer, steps, gradients, and checkpoints under
`artifacts/reports/`.
