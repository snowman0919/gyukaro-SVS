# GYU Renderer Protocol v2

Hybrid service stays resident:

```sh
gyu-singer --backend hybrid-svs serve --port 8765
```

`GET /health` returns readiness. `GET /model` returns backend, version, checkpoint, languages and 48 kHz rate. `POST /render` returns WAV without reloading the model.

Notes must use either seconds (`start`, `duration`) or beats (`start_beat`, `duration_beats`). Beat values convert as `seconds = beats * 60 / tempo` before phrase-frame alignment. Each note requires `id`, MIDI `pitch`, `lyric`; optional `slur` is accepted and preserved.

`curves.pitch` is a semitone residual curve consumed by `PitchConditionEncoder`; point form is `{ "beat": 1, "value": 0.5 }`, `{ "time": 0.5, "value": 0.5 }`, or `[beat, value]`. Other supported curve names are `dynamics`, `breathiness`, `tension`, `brightness`, `vibrato`; their phrase means condition `StyleEncoder`. `style.preset` accepts `neutral`, `soft`, `breathy`, `energetic`, `dark`, `bright`, `tense`, `vibrato`.

Unknown curves fail validation rather than silently doing nothing.
