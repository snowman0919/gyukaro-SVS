# GYU-SINGER for OpenUtau v0.9

This is an executable, maintained OpenUtau fork overlay. OpenUtau's current renderer API is internal to `OpenUtau.Core`; it has no external renderer-registration API, so a small pinned fork is the maintainable integration point. The overlay adds a built-in `GYU Singer` singer and a phrase renderer; its dummy OTO objects only let OpenUtau form phrases and never synthesize audio.

Pinned upstream: `stakira/OpenUtau` commit `27573ac5c888d927119d5f65a207312d79194b1f`.

## Install the fork

```sh
git clone https://github.com/stakira/OpenUtau.git
git -C OpenUtau checkout 27573ac5c888d927119d5f65a207312d79194b1f
./integrations/openutau/install_fork.sh OpenUtau
dotnet build OpenUtau/OpenUtau.csproj -c Release
```

Start the GPU-resident renderer once; models stay loaded across phrase requests:

```sh
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python  # optional when auto-discovery does not apply
./serve.sh
export GYU_RENDERER_URL=http://127.0.0.1:8765/render
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/model
```

Open `examples/openutau_v09.ustx` in the patched editor. Its KO, EN, and JA tracks already select the built-in `GYU-SINGER` singer and renderer. Playback/export uses multi-note `RenderPhrase` requests and the returned phrase WAV; there is no per-note TTS, waveform pitch shift, or phase-vocoder note control.

## Editor mapping

| OpenUtau data | GYU request |
|---|---|
| notes, tuning, lyrics | timed phrase notes |
| phrase tempo | `tempo` |
| generated phones | timed `phonemes` |
| note pitch, portamento, vibrato, PITD | authoritative `curves.pitch` residual |
| DYN, BREC, TENC | dynamics, breathiness, tension curves |
| GYUS 0..5 | neutral, soft, breathy, energetic, relative C, relative B |

OpenUtau's final `phrase.pitches` is sent, so note pitch, portamento, vibrato, and user PITD edits remain authoritative after learned GYU prosody. Phrase JSON SHA-256 keys the OpenUtau cache; relevant score, lyric, curve, or style edits invalidate it. Resident failures appear as a clean renderer exception pointing to `GYU_RENDERER_URL` and `/health`.

Only `breathy` and `energetic` have held-out acoustic-direction evidence. `soft`, `dark`, and `bright` remain relative controls and are not claimed to have calibrated perceptual semantics. Brightness, separate vibrato-depth controls, and other expressions not listed above are unsupported as independent curves.

`bridge.py` remains a headless export/debug utility. It supports tempo maps, selected phrase-relative timing, pitch/dynamics/breathiness/tension curves, and GYUS; it is not the native integration.

## Verification

```sh
dotnet test OpenUtau/OpenUtau.Test/OpenUtau.Test.csproj -c Release \
  --filter FullyQualifiedName~GyuSingerRendererTest
GYU_RENDERER_URL=http://127.0.0.1:8765/render \
  ./integrations/openutau/test_resident_fork.sh OpenUtau dotnet
PYTHONPATH=src python scripts/test_openutau_v09_behavior.py
```

The native tests load the three-language USTX and verify payload invalidation. The resident test executes the real C# `IRenderer.Render` path. The behavioral test verifies 48 kHz KO/EN/JA output, note-pitch and PITD F0 changes, Whisper-observed lyric changes, and latent energetic-style audio change.
