# SVS-05 Voicebank Factory Implementation Plan

1. Test and implement an affirmative rights/provenance manifest with no network acquisition.
2. Test and implement non-destructive audio inspection, normalization, corruption, duplicate, acoustic, and auxiliary language evidence.
3. Test transcript priority, untrusted Korean draft exclusion, and reviewed CSV/JSONL import.
4. Add energy VAD, phone generation, confidence classification, boundary evidence, and manual correction hooks.
5. Generate coverage and targeted recording plans; block training when insufficient.
6. Freeze adaptation strategy, hashes, deterministic splits, seeds, environment, early stop, and preservation-first checkpoint selection.
7. Implement train/evaluate/review/package gates without activating an unauthorized model.
8. Add phase state, structured logs, dry-run estimates, resume, idempotency, and orchestrated build stopping.
9. Run a synthetic, non-voice reproducible smoke and commit only its compact report.
10. Run full tests, dataset validation, package/CLI refusal smokes, evidence checks, and diff checks.
