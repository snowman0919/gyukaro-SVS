#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PORT="${GYU_SMOKE_PORT:-8765}"
OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-smoke}"
OPENUTAU_REPO="${OPENUTAU_REPO:-/tmp/OpenUtau}"

if [ -z "${GYU_SINGER_CACHE:-}" ]; then
  if [ -d "$ROOT_DIR/data/cache" ]; then
    GYU_SINGER_CACHE="$ROOT_DIR/data/cache"
  else
    echo "GYU_SINGER_CACHE is required (pin/cache root)" >&2
    exit 2
  fi
fi
if [ ! -d "$GYU_SINGER_CACHE" ] || [ ! -d "$GYU_SINGER_CACHE/omnivoice/.venv/bin" ]; then
  if [ "$GYU_SINGER_CACHE" != "$ROOT_DIR/data/cache" ] && [ -d "$ROOT_DIR/data/cache/omnivoice/.venv/bin" ]; then
    echo "GYU_SINGER_CACHE does not point to pinned cache; falling back to ROOT_DIR/data/cache" >&2
    GYU_SINGER_CACHE="$ROOT_DIR/data/cache"
  fi
fi
if [ ! -d "$GYU_SINGER_CACHE" ] || [ ! -d "$GYU_SINGER_CACHE/omnivoice/.venv/bin" ]; then
  echo "missing or invalid cache layout: $GYU_SINGER_CACHE" >&2
  exit 2
fi
if [ -z "${GYU_SOULX_PYTHON:-}" ]; then
  if [ -x "$ROOT_DIR/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT_DIR/.venv-soulx/bin/python"
  elif [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  else
    echo "GYU_SOULX_PYTHON is required and no auto-discover match was found" >&2
    exit 2
  fi
fi
if [ ! -x "$GYU_SOULX_PYTHON" ]; then
  echo "Invalid GYU_SOULX_PYTHON: $GYU_SOULX_PYTHON" >&2
  exit 2
fi

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"
: > "$OUTPUT_DIR/openutau_v09_smoke.log"

PYTHONPATH=src "$GYU_SOULX_PYTHON" -m gyu_singer.cli --backend gyu-singer-v0.8 --reference data/processed/master/216.wav serve --port "$PORT" >"$OUTPUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -sS "http://127.0.0.1:$PORT/health"
curl -sS "http://127.0.0.1:$PORT/model"

dotnet_bin=""
if [ -x /tmp/dotnet/dotnet ]; then
  dotnet_bin=/tmp/dotnet/dotnet
elif command -v dotnet >/dev/null 2>&1; then
  dotnet_bin=$(command -v dotnet)
fi

python "$ROOT_DIR/integrations/openutau/bridge.py" "$ROOT_DIR/examples/openutau_v09.ustx" \
  --language ko --part 0 --output "$OUTPUT_DIR/openutau_v09_request.json" \
  --render-url "http://127.0.0.1:$PORT" --wav "$OUTPUT_DIR/openutau_v09_smoke.wav"
python - <<PY
import hashlib, pathlib
p = pathlib.Path('$OUTPUT_DIR/openutau_v09_smoke.wav')
print('openutau_v09_smoke_wav_sha256', hashlib.sha256(p.read_bytes()).hexdigest())
PY

if [ -n "$dotnet_bin" ] && [ -d "$OPENUTAU_REPO" ] && [ -f "$OPENUTAU_REPO/OpenUtau.Test/OpenUtau.Test.csproj" ]; then
  GYU_RENDERER_URL="http://127.0.0.1:$PORT/render" "$dotnet_bin" test "$OPENUTAU_REPO/OpenUtau.Test/OpenUtau.Test.csproj" -c Release --filter "FullyQualifiedName~GyuSingerResidentIntegrationTest" >"$OUTPUT_DIR/openutau_resident_test.log" 2>&1
  tail -n 20 "$OUTPUT_DIR/openutau_resident_test.log"
fi

echo "openutau v0.9 runtime smoke done"
