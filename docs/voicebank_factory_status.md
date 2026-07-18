# Voicebank Factory Diagnostic Status

## Result

`dataset_needs_more_recording`

The implementation and synthetic smoke are complete. The smoke validates one local 44.1 kHz mono input, preserves its SHA-256, creates a 48 kHz mono workspace copy, resumes deterministically, blocks training for insufficient duration/phone/stress coverage, refuses release without human approval, and creates only a metadata diagnostic package.

This is infrastructure validation, not voice-model training or audio-quality evidence. No voice recordings, checkpoints, rendered WAVs, external datasets, or package archive are committed.

Reproduce with:

```sh
PYTHONPATH=src python scripts/smoke_voicebank_factory.py --check
```

Tracked evidence: `artifacts/reports/voicebank_factory/smoke.json`.
