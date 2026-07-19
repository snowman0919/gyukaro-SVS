# SVS-02 Experiment Framework Handoff

- Branch: `codex/svs-02-experiment-framework`
- Worktree: `/home/kotori9/code/gyukaro/.worktrees/svs-02-experiment-framework`
- Base: `18a085f8eef9d2577821406e94a7dc8107866035`
- Experiment implementation: `84b64d58ce647678a7543292669b19e14e93feea`
- Evidence registry: `5b961660720bb4618312f4fd6b476bcc75e4dcc0`
- Decision: `foundation_machine_inconclusive`
- Training: blocked, 0 optimizer steps
- Identity adaptation: not started
- Full test result: 162 passed, 9 warnings
- Dataset validation: 132 recordings, 106..237 sequential, 48 kHz mono PCM, corrupt 0
- Research evidence validation: 9 models, 9 evidence rows

## Tracked evidence

- `artifacts/reports/korean_phone_reassessment/evaluation.json`
- `artifacts/reports/korean_phone_reassessment/alignment_audit.json`
- `artifacts/reports/korean_phone_reassessment/human_review_manifest.json`
- `data/manifests/korean_alignment_audit.jsonl`
- `docs/korean_phone_reassessment.md`
- `docs/korean_linguistic_adapter.md`

## Local ignored evidence

`data/external/work/korean_phone_reassessment_review/` contains 21 files (6.9 MiB), including 15 blind WAV copies, plots, and a sealed local key. Source recordings, rendered WAVs, checkpoints, caches, and external datasets are not committed.

## Limitation and next step

MMS is target-conditioned and HuBERT measures seed consistency; neither is an independent calibrated Korean singing phone recognizer. The representation therefore remains unselected. The next stacked branch may enforce runtime/backend safety, but it must not train or select a Korean or identity adapter.
