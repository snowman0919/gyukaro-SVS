#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

: "${GYU_V09_PACKAGE_DIR=$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
: "${GYU_V09_READINESS_OUTPUT_DIR:=$ROOT/artifacts/reports/openutau_v09}"
: "${GYU_SMOKE_OUTPUT_DIR:=/tmp/gyu-v09-operational-check}"
: "${GYU_SMOKE_PORT:=8780}"
: "${GYU_SOULX_RUNTIME_DIR=}"
: "${GYU_SOULX_PYTHON=}"
: "${GYU_SINGER_CACHE=}"

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

if [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -n "${GYU_SOULX_RUNTIME_DIR:-}" ]; then
  if [ -x "$GYU_SOULX_RUNTIME_DIR/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/.venv/bin/python"
  elif [ -x "$GYU_SOULX_RUNTIME_DIR/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/bin/python"
  fi
fi

if [ -z "${GYU_SINGER_CACHE:-}" ]; then
  if [ -d "$ROOT/data/cache" ]; then
    GYU_SINGER_CACHE="$ROOT/data/cache"
  elif [ -d "$ROOT/../../data/cache" ]; then
    GYU_SINGER_CACHE="$ROOT/../../data/cache"
  fi
fi

export GYU_V09_PACKAGE_DIR
export GYU_V09_READINESS_OUTPUT_DIR
export GYU_SMOKE_OUTPUT_DIR
export GYU_SMOKE_PORT
export GYU_SOULX_RUNTIME_DIR
export GYU_SOULX_PYTHON
export GYU_SINGER_CACHE

printf '\n[OPS] step 1/3: dataset validation\n'
if [ -f "$ROOT/scripts/validate_dataset.py" ]; then
  if python "$ROOT/scripts/validate_dataset.py"; then
    echo "[OPS] dataset PASS"
  else
    echo "[OPS] dataset FAILED" >&2
    exit 2
  fi
else
  echo "[OPS] dataset SKIPPED (not available from package runtime path)"
fi

printf '\n[OPS] step 2/3: runtime readiness\n'
if "$SCRIPT_DIR/openutau_v09_full_runtime_readiness.sh"; then
  echo "[OPS] readiness PASS"
else
  echo "[OPS] readiness FAILED" >&2
  exit 2
fi

if [ "${GYU_OPS_RUN_PKG_TESTS:-0}" = "1" ]; then
  printf '\n[OPS] step 3/3: package smoke tests\n'
  if [ -f "$ROOT/tests/test_openutau_diffsinger_package.py" ] && \
     [ -f "$ROOT/tests/test_openutau_native_evaluation.py" ] && \
     [ -f "$ROOT/tests/test_hybrid.py" ]; then
    if python -m pytest tests/test_openutau_diffsinger_package.py tests/test_openutau_native_evaluation.py tests/test_hybrid.py::test_openutau_bridge_normalizes_render_url -q; then
      echo "[OPS] package tests PASS"
    else
      echo "[OPS] package tests FAILED" >&2
      exit 2
    fi
  else
    echo "[OPS] package tests SKIPPED (not available from package runtime path)"
  fi
else
  printf '\n[OPS] step 3/3: quick evidence copy\n'
  printf '  readiness: %s\n' "$GYU_V09_READINESS_OUTPUT_DIR/readiness_summary.json"
  printf '  behavior: %s\n' "$GYU_SMOKE_OUTPUT_DIR/openutau_v09_operational_behavior.json"
  printf '  smoke wav: %s\n' "$GYU_SMOKE_OUTPUT_DIR/openutau_v09_smoke.wav"
  echo "[OPS] skipped package tests; set GYU_OPS_RUN_PKG_TESTS=1 to enable"
fi

printf '\n[OPS] done\n'
