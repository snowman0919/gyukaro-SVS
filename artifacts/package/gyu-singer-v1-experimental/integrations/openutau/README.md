# OpenUtau USTX bridge

`bridge.py` parses UTF-8 YAML USTX, converts a selected `voice_parts` entry's tick timing and MIDI notes into GYU Renderer Protocol v2, then can POST it to resident renderer.

```sh
python integrations/openutau/bridge.py song.ustx --language ko --output song.json
gyu-singer serve --port 8765
python integrations/openutau/bridge.py song.ustx --language ko --output song.json --render-url http://127.0.0.1:8765 --wav song.wav
```

Bridge supports first tempo event only. Tempo maps, pitch curves, vibrato and native OpenUtau plugin registration remain future work.
