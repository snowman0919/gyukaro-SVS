# GYU Renderer Protocol v2

Quality service:

```sh
GYU_SINGER_CACHE=data/cache GYU_SOULX_PYTHON=.venv-soulx/bin/python \
PYTHONPATH=src python -m gyu_singer.cli --backend hybrid-soulx-phrase serve --port 8765
```

`GET /health` returns readiness. `GET /model` returns backend, version,
checkpoint, languages and 48 kHz rate. `POST /render` returns WAV.  The HTTP
server stays resident; ACE-Step and SoulX execute in their pinned runtimes for
each render, so this is functional editor integration, not a low-latency
resident-weight claim.

Notes must use either seconds (`start`, `duration`) or beats (`start_beat`, `duration_beats`). Beat values convert as `seconds = beats * 60 / tempo` before phrase-frame alignment. Each note requires `id`, MIDI `pitch`, `lyric`; optional `slur` is accepted and preserved.

`curves.pitch` is a semitone residual curve consumed by `PitchConditionEncoder`; point form is `{ "beat": 1, "value": 0.5 }`, `{ "time": 0.5, "value": 0.5 }`, or `[beat, value]`. Other supported curve names are `dynamics`, `breathiness`, `tension`, `brightness`, `vibrato`; their phrase means condition `StyleEncoder`. `style.preset` accepts `neutral`, `soft`, `breathy`, `energetic`, `dark`, `bright`, `tense`, `vibrato`.

Unknown curves fail validation rather than silently doing nothing.
