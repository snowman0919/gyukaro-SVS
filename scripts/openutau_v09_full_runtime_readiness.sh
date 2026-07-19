#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$ROOT/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-$ROOT/.venv-soulx/bin/python}"
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-$ROOT/data/cache}"
export GYU_V09_PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
export GYU_V09_READINESS_OUTPUT_DIR="${GYU_V09_READINESS_OUTPUT_DIR:-$ROOT/artifacts/reports/openutau_v09}"
export GYU_SMOKE_OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}"
export GYU_SMOKE_PORT="${GYU_SMOKE_PORT:-8780}"
export OPENUTAU_REPO="${OPENUTAU_REPO:-/tmp/OpenUtau}"

if [ ! -x "$GYU_SOULX_PYTHON" ]; then
  echo "invalid GYU_SOULX_PYTHON: $GYU_SOULX_PYTHON" >&2
  exit 2
fi
if [ ! -d "$GYU_SINGER_CACHE" ]; then
  echo "cannot find cache dir: $GYU_SINGER_CACHE" >&2
  exit 2
fi

cd "$ROOT"

echo "[FULL READY] step 1/3: scripted readiness check"
"$SCRIPT_DIR/openutau_v09_ready_check.sh"

echo "[FULL READY] step 2/3: resident+behavior operational check"
"$SCRIPT_DIR/openutau_v09_operational_check.sh" "$GYU_V09_PACKAGE_DIR"

echo "[FULL READY] step 3/3: OpenUtau package smoke"
export GYU_SMOKE_OUTPUT_DIR
cd "$GYU_V09_PACKAGE_DIR"
./scripts/openutau_v09_runtime_smoke.sh

echo "[FULL READY] done"
echo "readiness: $GYU_V09_READINESS_OUTPUT_DIR/readiness_summary.json"
echo "behavior : $GYU_SMOKE_OUTPUT_DIR/openutau_v09_operational_behavior.json"
echo "smoke wav: $GYU_SMOKE_OUTPUT_DIR/openutau_v09_smoke.wav"
