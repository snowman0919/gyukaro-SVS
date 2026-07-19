# GYU Singer

Personalized trilingual neural singing runtime for authorized GYU recordings.

## Quality path

The quality-tested path generates one duration-locked lyric phrase with
OmniVoice, then uses SoulX-Singer SVC with the complete score-derived 50 Hz F0 contour. It
does not use per-note TTS, pitch shifting, phase-vocoder timing, or waveform
concatenation.  It requires the locally cached upstream models and the pinned
SoulX Python environment:

```sh
export GYU_SINGER_CACHE="$PWD/data/cache"
export GYU_SOULX_PYTHON="$PWD/.venv-soulx/bin/python"  # optional if discoverable in cache
./serve.sh 8765
./render.sh examples/quality_ko.json gyu-ko.wav
```

`artifacts/reports/soulx_heldout_smoke.json` records the fixed held-out KO/EN/JA
quality gate: F0 correlation >= 0.90, pitch MAE <= 100 cents, held-note F0 CV
<= 0.10, and lyric similarity >= 0.50.  The package builder is
`scripts/package_quality_runtime.py`; it intentionally excludes upstream model caches.

## Renderer API and OpenUtau

```sh
./serve.sh 8765
export GYU_SINGER_CACHE="$PWD/data/cache"
export GYU_SOULX_PYTHON="$PWD/.venv-soulx/bin/python"
./serve.sh 8765
curl -s http://127.0.0.1:8765/health
python integrations/openutau/bridge.py examples/openutau_smoke.ustx --language ko \
  --output song.json --render-url http://127.0.0.1:8765 --wav song.wav
```

## Experimental compact model

`--backend hybrid-compact-experimental` remains an inspectable phrase-level TriSinger model,
with score reconstruction, forced alignment, score-only pitch input, teacher
distillation, blurred boundaries, and residual flow.  Its generated quality
fails the acceptance gate.  Do not use it as the quality candidate.  Evidence
and limitations: `docs/evaluation_v2_report.md` and `docs/final_v2_report.md`.
