#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
if [ -d "$ROOT/artifacts/package/gyu-singer-v0.9-openutau" ]; then
  DEFAULT_PACKAGE_DIR="$ROOT/artifacts/package/gyu-singer-v0.9-openutau"
elif [ -f "$ROOT/serve.sh" ] && [ -d "$ROOT/scripts" ] && [ -f "$ROOT/integrations/openutau/test_resident_fork.sh" ]; then
  DEFAULT_PACKAGE_DIR="$ROOT"
else
  DEFAULT_PACKAGE_DIR="$ROOT/artifacts/package/gyu-singer-v0.9-openutau"
fi

if [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/.venv-soulx/.venv/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../.venv-soulx/.venv/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../../.venv-soulx/.venv/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../../.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../../../.venv-soulx/.venv/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../../../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../../../.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../../../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$HOME/.venv-soulx/.venv/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$HOME/.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$HOME/.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$HOME/.venv-soulx"
fi

if [ -z "${GYU_SOULX_PYTHON:-}" ]; then
  if [ -n "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$GYU_SOULX_RUNTIME_DIR/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/.venv/bin/python"
  elif [ -n "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$GYU_SOULX_RUNTIME_DIR/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/bin/python"
  elif [ -x "$ROOT/.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/.venv-soulx/bin/python"
  elif [ -x "$ROOT/../.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/../.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../.venv-soulx/bin/python"
  elif [ -x "$ROOT/../../.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/../../.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../.venv-soulx/bin/python"
  elif [ -x "$ROOT/../../../.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../../.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/../../../.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../../.venv-soulx/bin/python"
  elif [ -x "$HOME/.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/.venv/bin/python"
  elif [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  fi
fi

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$ROOT/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-$ROOT/.venv-soulx/bin/python}"
if [ -z "${GYU_SINGER_CACHE:-}" ] && [ -d "$ROOT/data/cache" ]; then
  GYU_SINGER_CACHE="$ROOT/data/cache"
elif [ -z "${GYU_SINGER_CACHE:-}" ] && [ -d "$ROOT/../../data/cache" ]; then
  GYU_SINGER_CACHE="$ROOT/../../data/cache"
elif [ -z "${GYU_SINGER_CACHE:-}" ] && [ -d "$HOME/code/gyukaro/data/cache" ]; then
  GYU_SINGER_CACHE="$HOME/code/gyukaro/data/cache"
fi
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-$ROOT/data/cache}"
export GYU_V09_PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$DEFAULT_PACKAGE_DIR}"
export GYU_V09_READINESS_OUTPUT_DIR="${GYU_V09_READINESS_OUTPUT_DIR:-$ROOT/artifacts/reports/openutau_v09}"
export GYU_SMOKE_OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}"
export GYU_SMOKE_PORT="${GYU_SMOKE_PORT:-8780}"
export OPENUTAU_REPO="${OPENUTAU_REPO:-/tmp/OpenUtau}"
export GYU_OPS_RUN_PKG_TESTS="${GYU_OPS_RUN_PKG_TESTS:-1}"

echo "[PRODUCTION READINESS] fixed runtime path gate"
echo "ROOT: $ROOT"
echo "PACKAGE_DIR: $GYU_V09_PACKAGE_DIR"
echo "CACHE: $GYU_SINGER_CACHE"
echo "PYTHON: $GYU_SOULX_PYTHON"
echo "PORT: $GYU_SMOKE_PORT"
echo "OPENUTAU_REPO: $OPENUTAU_REPO"

"$SCRIPT_DIR/openutau_v09_ops_check.sh"
