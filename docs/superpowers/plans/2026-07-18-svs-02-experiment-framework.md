# SVS-02 Experiment Framework Implementation Plan

1. Freeze Korean probe cases, three phone representations, seeds, and evaluator roles.
2. Add deterministic representation and aggregation tests before implementation.
3. Audit existing Korean alignment manifests and label every non-manual row as inferred.
4. Calibrate target-conditioned MMS alignment on script-known GYU recordings and mismatched controls.
5. Evaluate existing GTSinger soprano Korean renders with MMS, HuBERT, pitch, voicing, and artifact metrics.
6. Apply the frozen decision policy without using Whisper as a primary gate.
7. Add a zero-equivalent frozen-foundation adapter scaffold, but do not optimize without a selected representation.
8. Build a blind seed-stability review package without transcript hints.
9. Run focused and complete tests, dataset validation, evidence validation, and `git diff --check`.
10. Commit compact reproducible evidence; keep WAVs, checkpoints, datasets, and caches ignored.
