#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PACKAGE_DIR="${1:-$SCRIPT_DIR/artifacts/package/gyu-singer-v0.9-openutau}"
PORT="${GYU_SMOKE_PORT:-8780}"
OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}"
mkdir -p "$OUTPUT_DIR"

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$SCRIPT_DIR/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-}"
if [ -z "$GYU_SOULX_PYTHON" ]; then
  if [ -x "$GYU_SOULX_RUNTIME_DIR/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/.venv/bin/python"
  elif [ -x "$GYU_SOULX_RUNTIME_DIR/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/bin/python"
  elif [ -x "$SCRIPT_DIR/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$SCRIPT_DIR/.venv-soulx/bin/python"
  elif [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  fi
  export GYU_SOULX_PYTHON
fi

if [ -z "${GYU_SINGER_CACHE:-}" ]; then
  if [ -d "$SCRIPT_DIR/data/cache" ]; then
    GYU_SINGER_CACHE="$SCRIPT_DIR/data/cache"
  elif [ -d "$HOME/code/gyukaro/data/cache" ]; then
    GYU_SINGER_CACHE="$HOME/code/gyukaro/data/cache"
  fi
  export GYU_SINGER_CACHE
fi

: "${GYU_SOULX_RUNTIME_DIR:?set GYU_SOULX_RUNTIME_DIR to pinned runtime}"
: "${GYU_SOULX_PYTHON:?set GYU_SOULX_PYTHON or provide valid GYU_SOULX_RUNTIME_DIR}"
: "${GYU_SINGER_CACHE:?set GYU_SINGER_CACHE}"
if [ ! -x "$GYU_SOULX_PYTHON" ]; then
  echo "invalid GYU_SOULX_PYTHON: $GYU_SOULX_PYTHON" >&2
  exit 2
fi
if [ ! -d "$PACKAGE_DIR" ]; then
  echo "cannot find package dir: $PACKAGE_DIR" >&2
  exit 2
fi

trap 'if [ -n "${SERVE_PID:-}" ] && kill -0 "$SERVE_PID" >/dev/null 2>&1; then kill "$SERVE_PID" >/dev/null 2>&1 || true; fi' EXIT

printf '\n[1/3] fixed-path smoke (package source path)...\n' >&2
cd "$SCRIPT_DIR"
if ! GYU_SOULX_RUNTIME_DIR="$GYU_SOULX_RUNTIME_DIR" GYU_SINGER_CACHE="$GYU_SINGER_CACHE" ./scripts/verify_v09_runtime_paths.sh "$PACKAGE_DIR"; then
  echo "verify_v09_runtime_paths failed" >&2
  exit 2
fi

printf '\n[2/3] start runtime and run phrase-level behavior check...\n' >&2
cd "$PACKAGE_DIR"
setsid env GYU_SOULX_RUNTIME_DIR="$GYU_SOULX_RUNTIME_DIR" GYU_SOULX_PYTHON="$GYU_SOULX_PYTHON" GYU_SINGER_CACHE="$GYU_SINGER_CACHE" \
  ./serve.sh "$PORT" >/tmp/openutau_v09_operational_serve.log 2>&1 &
SERVE_PID=$!

ready=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "runtime health check failed on port $PORT" >&2
  tail -n 40 /tmp/openutau_v09_operational_serve.log >&2
  exit 2
fi

printf '\n[3/3] render behavior output...\n' >&2
cd "$SCRIPT_DIR"
REPORT_PATH="$OUTPUT_DIR/openutau_v09_operational_behavior.json"
LOG_PATH="$OUTPUT_DIR/openutau_v09_operational_behavior.log"
if ! PYTHONPATH=src "$GYU_SOULX_PYTHON" "$PACKAGE_DIR/scripts/test_openutau_v09_behavior.py" \
  --render-url "http://127.0.0.1:$PORT" \
  --output "$REPORT_PATH" \
  >"$LOG_PATH" 2>&1; then
  echo "behavior script failed; tail of $LOG_PATH:" >&2
  tail -n 80 "$LOG_PATH" >&2
  exit 2
fi

cat "$REPORT_PATH"
