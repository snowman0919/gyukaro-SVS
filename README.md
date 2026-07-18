# GYU Singer Research Stack

**No production-approved singing model currently exists.**
**OpenUtau and release paths remain blocked.**

This repository contains research evidence, bounded experiment tooling, and
experimental renderers for authorized GYU recordings. Model status is defined
by `configs/project_status.json`; model names and successful WAV generation do
not imply release quality.

## Strongest validated capability

The GTSinger Japanese soprano foundation passed the historical Japanese
held-out matrix under the recorded evaluation protocol. It is foundation-only,
not a GYU singer. Korean lexical validity requires reassessment with
phone-centered evaluation; Korean Whisper is auxiliary evidence only.

The former OmniVoice-to-SoulX phrase path, RC8 candidate 3, truncated K=2/K=4,
GTSinger tenor, and GYU mix20 paths are rejected. RC7 is retained only as an
accepted experimental baseline.

## Explicit experimental rendering

Rejected and experimental backends are not quality paths. Historical commands
remain available for reproducing evidence with local caches, but must not be
used to claim production or OpenUtau readiness:

```sh
GYU_SINGER_CACHE="$PWD/data/cache" \
GYU_SOULX_PYTHON="$PWD/.venv-soulx/bin/python" \
PYTHONPATH=src python -m gyu_singer.cli --backend hybrid-svs --allow-experimental \
  --reference data/processed/master/216.wav \
  render examples/quality_ko.json --output gyu-ko.wav
```

Status and evidence hashes are recorded in `configs/project_status.json` and
`configs/research_evidence.json`.

## Historical renderer API

```sh
GYU_SINGER_CACHE="$PWD/data/cache" GYU_SOULX_PYTHON="$PWD/.venv-soulx/bin/python" \
PYTHONPATH=src python -m gyu_singer.cli --backend hybrid-svs --allow-experimental serve --port 8765
python integrations/openutau/bridge.py examples/openutau_smoke.ustx --language ko \
  --output song.json --render-url http://127.0.0.1:8765 --wav song.wav
```

The bridge above is diagnostic infrastructure, not an approved voicebank path.

With no production-approved backend, `render` and `serve` fail closed when
`--backend` is omitted. Every current backend requires the explicit diagnostic
override and writes an `EXPERIMENTAL_OVERRIDE` record to stderr.

The central release gate currently fails all 11 required dimensions. The safe
OpenUtau packager refuses release output. It can emit metadata-only evidence
with `scripts/package_openutau_safe.py --diagnostic-package` only when the
output directory name ends in `-diagnostic`; such output is marked
`NOT A RELEASE` and contains no checkpoint or audio.

## Experimental compact model

`--backend hybrid-compact-experimental` remains an inspectable phrase-level TriSinger model,
with score reconstruction, forced alignment, score-only pitch input, teacher
distillation, blurred boundaries, and residual flow.  Its generated quality
fails the acceptance gate.  Do not use it as the quality candidate.  Evidence
and limitations: `docs/evaluation_v2_report.md` and `docs/final_v2_report.md`.
