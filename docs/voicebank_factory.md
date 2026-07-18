# Automated Voicebank Factory

## Commands

```text
gyu-voicebank init --input DIR --name NAME --languages ko,ja,en --workspace DIR --rights-manifest FILE [--dry-run]
gyu-voicebank inspect --workspace DIR
gyu-voicebank prepare --workspace DIR [--review-manifest FILE]
gyu-voicebank train --workspace DIR
gyu-voicebank evaluate --workspace DIR
gyu-voicebank review-pack --workspace DIR
gyu-voicebank package --workspace DIR (--diagnostic | --release)
gyu-voicebank build --workspace DIR
gyu-voicebank status --workspace DIR
```

Use `configs/voicebank_rights.template.json`; `permission_affirmed` must be true. The factory does not download recordings. Source files are only read. Normalized audio and all private manifests stay in the user-selected workspace.

`prepare` exports `manifests/transcript_review.csv`. Accepted corrections can be re-imported with `--review-manifest`; unreviewed automatic drafts are absent from `frozen_dataset.jsonl`.

`reports/coverage.json`, `recording_plan.json`, and `adaptation_plan.json` explain why training is allowed or blocked. `state.json` and `factory.jsonl` support resume and failure diagnosis. Diagnostic packaging contains metadata only and is marked `NOT A RELEASE`. Release mode requires both a local approved human record and the central release engine.

The repository smoke uses a generated sine wave, not a voice or copyrighted material. It stops at `dataset_needs_more_recording`, refuses training and release, and commits no audio.
