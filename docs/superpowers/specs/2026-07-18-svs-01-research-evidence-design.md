# SVS Phase 0 Research Evidence Design

## Status

Approved by the user-provided Phase 0–5 execution goal. This branch is a
normalization layer only; it does not promote, train, render, or package a
model.

## Design

`configs/project_status.json` is the sole machine-readable model-status
registry. It separates software version, model family, experiment identity,
release state, and human approval. Existing reports remain immutable evidence;
`configs/research_evidence.json` binds their paths to hashes and originating
commits. `scripts/check_research_evidence.py` validates both files without
loading model frameworks.

`gyu_singer.paths.project_roots()` resolves project, data, cache, and evidence
roots from `GYUKARO_ROOT`, `GYUKARO_DATA_ROOT`, `GYUKARO_CACHE_ROOT`, and
`GYUKARO_EVIDENCE_ROOT`, with repository-relative defaults. Only newly touched
machine-specific paths are migrated; historical experiment configs remain
evidence and are not blindly rewritten.

The README must say that no production-approved model exists. Python support is
the actually tested 3.11 series. Base project dependencies are the `core`
runtime group; optional evaluation, DiffSinger, training, OpenUtau, factory,
and development groups are named in `pyproject.toml`. Binary training and
render outputs stay ignored.

## Gates

- Korean soprano status is `foundation_ko_lexical_unverified`, not passed or rejected.
- Rejected SoulX, RC8, truncated K=2/K=4, tenor, and mix20 states cannot drift.
- Japanese soprano remains `foundation_only`.
- Pure registry/path tests import without Torch or Transformers.
- Full tests, evidence validation, dataset validation, and `git diff --check` pass.
