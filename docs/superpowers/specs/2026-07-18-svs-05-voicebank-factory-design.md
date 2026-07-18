# SVS-05 Automated Voicebank Factory Design

## Goal

Create a resumable local project from authorized recordings without overwriting source audio, inventing transcript truth, bypassing model gates, or producing an unapproved release.

## State machine

`init → inspect → prepare → train → evaluate → review-pack → package`

Every completed phase atomically updates `state.json` and appends a structured event to `factory.jsonl`. `build` resumes from state and returns the specific unmet gate. Rerunning a completed compatible init or a failed coverage build is idempotent.

## Trust and data flow

Init requires an affirmative rights manifest. Inspection reads local WAV/FLAC/OGG/AIFF, reports corruption, exact and heuristic near duplicates, sample format, duration, silence, clipping, DC, noise, pitch estimate, auxiliary language estimate, and heuristic acoustic outliers. Prepare writes new 48 kHz mono PCM files under the workspace; source files remain unchanged.

Transcript trust order is exact user script, verified metadata, accepted user correction, then automatic draft. Korean automatic text is `untrusted_draft_transcript` and excluded until review. Editable CSV/JSONL corrections are importable.

Energy VAD, language-specific phone generation, confidence-classified alignment hooks, boundary JSON, and manual correction fields are produced. Uniform inferred alignment is never labeled manual.

## Gates

Coverage includes phones, language counts, duration, pitch range, rapid transitions, sustain, large intervals, phrase boundaries, voiced/unvoiced balance, and recording conditions. Missing mandatory coverage returns `dataset_needs_more_recording` plus targeted prompts.

The adaptation planner never selects the rejected SoulX route. Dataset splits and seeds are frozen before training; duplicate audio hashes are rejected. Checkpoint selection is preservation-first and deterministic. Current foundations do not authorize a complete multilingual candidate, so the smoke path does not train.

Evaluation policies are language-aware. Korean is phone-centered with Whisper auxiliary; JA/EN ASR is also auxiliary. Release packaging requires an approved human record and the central release gate. Without approval, the ceiling is `voicebank_candidate_human_pending`; the current smoke remains below it.
