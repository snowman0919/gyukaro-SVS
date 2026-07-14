# GYU Singer

Personalized trilingual neural singing runtime for authorized GYU recordings.

## Quality path

The quality-tested path generates one whole lyric phrase with ACE-Step, then
uses SoulX-Singer SVC with the complete score-derived 50 Hz F0 contour.  It
does not use per-note TTS, pitch shifting, phase-vocoder timing, or waveform
concatenation.  It requires the locally cached upstream models and the pinned
SoulX Python environment:

```sh
GYU_SINGER_CACHE="$PWD/data/cache" \
GYU_SOULX_PYTHON="$PWD/.venv-soulx/bin/python" \
PYTHONPATH=src python -m gyu_singer.cli --backend hybrid-soulx-phrase \
  --reference data/processed/master/216.wav \
  render examples/quality_ko.json --output gyu-ko.wav
```

`artifacts/reports/soulx_runtime_smoke.json` records the fixed KO/EN/JA
quality gate: F0 correlation >= 0.90, pitch MAE <= 100 cents, held-note F0 CV
<= 0.10, and lyric similarity >= 0.50.  The package builder is
`scripts/package_quality_runtime.py`; it intentionally excludes the 13 GB
upstream model cache.

## Renderer API and OpenUtau

```sh
GYU_SINGER_CACHE="$PWD/data/cache" GYU_SOULX_PYTHON="$PWD/.venv-soulx/bin/python" \
PYTHONPATH=src python -m gyu_singer.cli --backend hybrid-soulx-phrase serve --port 8765
python integrations/openutau/bridge.py examples/openutau_smoke.ustx --language ko \
  --output song.json --render-url http://127.0.0.1:8765 --wav song.wav
```

## Experimental compact model

`--backend hybrid-svs` remains an inspectable phrase-level TriSinger model,
with score reconstruction, forced alignment, score-only pitch input, teacher
distillation, blurred boundaries, and residual flow.  Its generated quality
fails the acceptance gate.  Do not use it as the quality candidate.  Evidence
and limitations: `docs/evaluation_v2_report.md` and `docs/final_v2_report.md`.
