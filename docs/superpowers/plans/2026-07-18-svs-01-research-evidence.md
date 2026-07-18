# SVS Phase 0 Research Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make current model status, evidence provenance, dependency boundaries, paths, and binary policy unambiguous without changing a renderer.

**Architecture:** Two stdlib-only modules resolve roots and validate static JSON registries. Existing reports remain immutable inputs; a thin script checks their content hashes.

**Tech Stack:** Python 3.11 stdlib, JSON, pytest, setuptools/pyproject.

## Global Constraints

- No production-approved singing model currently exists.
- Korean Whisper is auxiliary evidence only.
- Source recordings, rendered corpora, checkpoints, caches, and external datasets remain uncommitted.
- Do not modify runtime selection, package generation, or OpenUtau behavior on this branch.

---

### Task 1: Portable roots and evidence registries

**Files:**
- Create: `src/gyu_singer/paths.py`
- Create: `src/gyu_singer/research.py`
- Create: `configs/project_status.json`
- Create: `configs/research_evidence.json`
- Create: `tests/test_research_evidence.py`

**Interfaces:**
- Produces: `project_roots()`, `validate_status_registry(dict)`, `validate_evidence_manifest(dict, Path)`.

- [x] Write tests for root overrides, status separation, rejection preservation, hashes, and optional-import isolation.
- [x] Run `pytest tests/test_research_evidence.py -q` and observe missing-module failure.
- [x] Implement the two stdlib-only modules and registries.
- [x] Run the focused test and require all cases to pass.

### Task 2: Claims, dependencies, and artifact policy

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `scripts/run_gtsinger_gyu_identity_diagnostic.py`
- Create: `docs/artifact_policy.md`

**Interfaces:**
- Consumes: environment root variables from Task 1.
- Produces: evidence-correct public claims and Python 3.11 dependency groups.

- [x] Add failing assertions for README claims, Python range, dependency groups, binary ignores, and absence of the touched absolute home path.
- [x] Correct the README and pyproject metadata.
- [x] Extend binary ignores and replace only the touched diagnostic fallback path.
- [x] Run the focused tests.

### Task 3: Reproducible check and branch handoff

**Files:**
- Create: `scripts/check_research_evidence.py`
- Create: `docs/handoffs/svs-01-research-evidence.md`

**Interfaces:**
- Consumes: both JSON registries.
- Produces: zero-model-load validation command and branch handoff.

- [x] Add a failing CLI smoke test.
- [x] Implement the thin check script.
- [ ] Run full pytest, `scripts/check_research_evidence.py`, dataset validation, and `git diff --check`.
- [ ] Commit focused changes and record the final HEAD in the handoff.
