# SVS-04 Release and OpenUtau Safety Design

## Goal

Centralize release decisions and make it impossible for rejected, blocked, foundation-only, machine-inconclusive, or human-pending work to become an OpenUtau release package by accident.

## Release decision

The engine requires all 11 frozen dimensions: approved foundation, approved identity, phone-centered lexical evidence, score/pitch, voicing, artifacts, multi-seed stability, long-form continuity, recorded human approval, license/provenance, and a reproducible package manifest. Missing or unknown gates are invalid. Whisper is auxiliary and cannot be the sole lexical method.

## Packaging policy

Normal packaging additionally requires a `production_approved` backend whose central registry explicitly allows package and OpenUtau use. Refusal occurs before creating the output directory.

Diagnostic mode is metadata-only. Its output name must end in `-diagnostic`; `PACKAGE.json` and README begin with `NOT A RELEASE`, model activation is disabled, and checkpoints, source recordings, samples, WAVs, and datasets are absent. File hashes are deterministic for a fixed commit and protocol.

The current state is `release_blocked`. No release or OpenUtau package is authorized.
