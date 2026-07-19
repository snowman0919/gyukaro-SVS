# SVS-03 Runtime Safety Design

## Goal

Prevent a rejected or blocked renderer from being selected as a normal user path while preserving explicit diagnostic reproduction. Prepare a bounded GYU identity experiment protocol without authorizing training or runtime integration.

## Runtime policy

`configs/backend_registry.json` is authoritative for every CLI backend. Only `production_approved` may run without an override. With no approved backend, render and serve fail clearly instead of selecting `hybrid-svs`. Non-production execution requires `--allow-experimental` and emits a structured audit record.

The registry and implemented renderer names must match exactly. Linked project models must have the same status. Package and OpenUtau flags must be false for every non-production row.

## Identity protocol

The frozen comparison order is fixed GYU embedding, zero-equivalent FiLM, zero-equivalent low-rank residual, then optional vocoder conditioning. Vocoder conditioning remains disabled. Phone lexical, duration, pitch, voicing, artifact, seed-stability, and Japanese-foundation preservation precede both WavLM and ECAPA identity evidence.

Training requires an authorized Korean foundation and recorded human approval. The current `foundation_machine_inconclusive` state satisfies neither condition, so optimizer steps remain zero and runtime integration remains false.
