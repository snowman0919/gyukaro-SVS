# Renderer protocol v1

`gyu-singer --model checkpoints/gyu_v1_experimental.npz render input.json --output output.wav`

Input has `sample_rate`, `language`, `tempo`, and ordered `notes`. Each note needs `pitch` (MIDI), `start`, `duration`, and `lyric`; optional `dynamics` is 0..1. `gyu-singer serve --port 8765` keeps loops resident and accepts JSON `POST /render`, returning WAV.
