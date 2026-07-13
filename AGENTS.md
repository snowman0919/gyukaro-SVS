# GYU Singer rules

- Keep original recordings unchanged under `data/source/`; never commit them.
- Generated manifests and reports are reproducible from `scripts/`.
- Mark inferred labels as inferred. Do not present source-loop renderer as neural SVS.
- Run `python scripts/validate_dataset.py` and package smoke test after runtime changes.
