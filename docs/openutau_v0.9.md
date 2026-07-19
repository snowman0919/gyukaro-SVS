# OpenUtau v0.9 integration

## Architecture

The supported integration is a small maintained fork overlay against current official OpenUtau commit `27573ac5c888d927119d5f65a207312d79194b1f`. Current OpenUtau exposes `IRenderer` and renderer registration inside `OpenUtau.Core`, but no external renderer-registration API. A pinned three-file registration patch plus the self-contained `GyuSingerRenderer.cs` is therefore smaller and more stable than pretending a standalone exporter is native integration.

`GYU-SINGER` is a built-in virtual singer. Its dummy OTO entries only allow normal `RenderPhrase` construction; they emit no audio. OpenUtau sends each multi-note phrase to a single resident `gyu-singer-v0.8` process, receives one 48 kHz WAV, performs the single editor-standard 44.1 kHz conversion, then caches and plays/exports the phrase.

```text
OpenUtau RenderPhrase
  -> notes/lyrics/phones/tempo/final pitch and expression curves
  -> GYU protocol v2 JSON
  -> resident v0.8 (v0.5 prosody + v0.7 identity/style + SoulX)
  -> 48 kHz phrase WAV
  -> OpenUtau cache/playback/export
```

The static HTTP client and resident workers avoid reloading OmniVoice or SoulX per phrase. SHA-256 of the complete request is the cache key, so note, lyric, pitch, expression, and style edits invalidate the cached WAV. Network/model errors are surfaced as a renderer failure naming `GYU_RENDERER_URL` and `/health`.

## Mapped controls

- Notes, tuning, lyrics, tempo, and generated phonemes are timed from the actual `RenderPhrase`.
- OpenUtau `phrase.pitches` already combines note pitch, portamento, note vibrato, and PITD. It is converted to a score-relative semitone residual and remains authoritative after learned GYU prosody.
- DYN, BREC, and TENC map to dynamics, breathiness, and tension.
- GYUS values 0..5 map to neutral, soft, breathy, energetic, relative C, and relative B.
- Unsupported as separate controls: brightness curve, an independent vibrato-depth curve, and expressions not listed above.

Only breathy and energetic have repeated acoustic-proxy direction evidence. Soft, dark, and bright are relative latent controls; no calibrated semantic claim is made.

## Evidence

The overlay compiles on .NET 8 against the pinned official tree. The native package checks run through 4 tests (package checks + OpenUtau `OpenUtau.Test` mapping integration), and a separate resident integration test executes the real C# `IRenderer.Render` call and receives non-silent 44.1 kHz phrase samples from the running v0.8 backend.

The reproducible behavioral report is `artifacts/reports/openutau_v09/behavior.json`:

- KO/EN/JA: 48 kHz mono, 2.12 seconds each.
- +2-semitone note edit: RMVPE median shift +200.41 cents.
- +1-semitone PITD edit: RMVPE median shift +92.52 cents.
- lyric edit: Whisper changes from “하늘빛 노래” to “사랑을 담아서 내준 내 마음”; waveform RMS difference 0.118047.
- neutral vs energetic: final RMS 0.078379 vs 0.079411 with score, lyrics, identity, and pitch fixed.

This proves causal editor behavior, not a listening-quality preference. The energetic effect is small but nonzero and agrees with the prior held-out energy proxy direction.

## Install and test

See `integrations/openutau/README.md`. The essential commands are:

```sh
git clone https://github.com/stakira/OpenUtau.git
git -C OpenUtau checkout 27573ac5c888d927119d5f65a207312d79194b1f
./integrations/openutau/install_fork.sh OpenUtau
dotnet build OpenUtau/OpenUtau.csproj -c Release
# If you run from repository source checkout root, use <repo-root>/data/cache
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python  # optional when auto-discovery does not apply
# or
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-root  # optional: directory containing .venv/bin/python or bin/python
export GYU_RENDERER_URL=http://127.0.0.1:8765/render
./serve.sh 8765
```

For source-tree developers (before packaging), start the renderer with:

```sh
./serve.sh 8765
```

Note: if `dotnet` is not on PATH, these helpers can still run with local runtimes
such as `/tmp/dotnet/dotnet`.

Then open `examples/openutau_v09.ustx`. All three tracks use the phrase renderer. The headless `bridge.py` remains only a debugging/export path and is not counted as the v0.9 integration.

## Runtime smoke

For repeatable smoke checks (recommended), run:

```sh
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
# If you run from repository source checkout root, use <repo-root>/data/cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python
# or
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-root
export OPENUTAU_REPO=/absolute/path/to/patched/OpenUtau
export GYU_SMOKE_PORT=8765
export GYU_SMOKE_OUTPUT_DIR=/tmp/gyu-v09-smoke
./scripts/openutau_v09_runtime_smoke.sh
```

The script runs: resident boot, `/health`, `/model`, bridge render, and resident integration test.

For a single fixed-runtime-path check (source-tree and packaged tree), use:

```sh
scripts/verify_v09_runtime_paths.sh /tmp/gyu-singer-v0.9-openutau
```

Equivalent manual flow:

```sh
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python
# or
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-root
./serve.sh 8765 >/tmp/gyu-singer-v0.9-serve.log 2>&1 &
sleep 2
curl -s http://127.0.0.1:8765/health
python integrations/openutau/bridge.py examples/openutau_v09.ustx --language ko \
  --output /tmp/openutau-v09-request.json --render-url http://127.0.0.1:8765 --wav /tmp/openutau-v09-smoke.wav

cd /tmp/OpenUtau
GYU_RENDERER_URL=http://127.0.0.1:8765/render dotnet test OpenUtau.Test/OpenUtau.Test.csproj -c Release --filter FullyQualifiedName~GyuSingerResidentIntegrationTest

cd /path/to/gyukaro
python -m pytest tests/test_openutau_diffsinger_package.py tests/test_openutau_native_evaluation.py
```
