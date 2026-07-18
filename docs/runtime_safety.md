# Runtime Safety Status

## Result

`runtime_safety_complete`

There is no production-approved backend. Normal `render` and `serve` commands fail closed. All 14 historical or diagnostic CLI backends are rejected or blocked and require `--allow-experimental`; the override is logged to stderr with backend, status, and reason.

RC8 remains rejected. RC9 remains blocked. Package and OpenUtau selection are false for all non-production registry rows. No default checkpoint or renderer changed.

The machine-readable result is `artifacts/reports/runtime_safety/status.json`, reproducible with:

```sh
PYTHONPATH=src python scripts/check_runtime_safety.py --check
```
