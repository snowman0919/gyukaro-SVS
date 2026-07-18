# Binary and private artifact policy

Commit code, tests, compact JSON/CSV summaries, hashes, configuration, and
small non-audio fixtures. Do not commit source recordings, processed private
audio, external datasets, rendered WAV corpora, checkpoints, model caches,
temporary plots/arrays, or package builds.

Default local roots are `data/`, `data/cache/`, and `artifacts/reports/`.
Override them with `GYUKARO_DATA_ROOT`, `GYUKARO_CACHE_ROOT`, and
`GYUKARO_EVIDENCE_ROOT`. Tools must never overwrite `data/source/`.

Evidence manifests may reference ignored local files only with
`"local_only": true`; tracked evidence must include a verified SHA-256.
