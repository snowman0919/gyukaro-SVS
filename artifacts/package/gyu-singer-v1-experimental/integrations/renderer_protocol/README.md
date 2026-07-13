# GYU Renderer Protocol v2

Resident server command:

```sh
gyu-singer --backend hybrid-svs serve --port 8765
```

`GET /health` returns readiness. `GET /model` returns backend and output rate. `POST /render` accepts JSON and returns a 48 kHz PCM WAV without reloading model.

Required score fields: `language` (`ko`, `en`, `ja`) and ordered non-overlapping `notes`. Every note has MIDI `pitch`, second-based `start` and `duration`, and `lyric`. Optional `expressions`: scalar `dynamics`, `breathiness`, `tension`, `brightness`, `vibrato`.

`integrations/openutau/bridge.py` converts one USTX voice part to this protocol.
