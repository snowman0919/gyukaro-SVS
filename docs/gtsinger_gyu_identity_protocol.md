# Bounded GTSinger-to-GYU Identity Protocol

## Status

`identity_machine_inconclusive`

Training authorization: false. Optimizer steps: 0. Runtime integration: false.

The frozen candidate order is:

1. fixed GYU speaker embedding
2. small zero-equivalent FiLM
3. zero-equivalent low-rank residual
4. optional vocoder conditioning, currently disabled

References 212, 215, 216, 219, and 220 and seeds 7, 21, and 42 are fixed before any future authorized run. Both WavLM and ECAPA must improve consistently, but identity cannot compensate for phone lexical, duration, pitch, voicing, clipping, HF-spike, sample-jump, seed-stability, or Japanese-foundation regression.

The current Korean foundation is `foundation_machine_inconclusive`, not phone-validated or human-approved. The authorization check therefore blocks training before optimizer initialization. The adapter classes are experiment-only and are not imported by the CLI, renderer, package, or OpenUtau path.
