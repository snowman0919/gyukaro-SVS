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
export OPENUTAU_REPO="${OPENUTAU_REPO:-/tmp/OpenUtau}"
export GYU_SMOKE_PORT="${GYU_SMOKE_PORT:-8780}"
export GYU_OPS_RUN_PKG_TESTS=1

cd "$ROOT"

printf '%s\n' "[ONECLICK] fixed runtime environment:"
printf '  GYU_SOULX_RUNTIME_DIR=%s\n' "$GYU_SOULX_RUNTIME_DIR"
printf '  GYU_SOULX_PYTHON=%s\n' "$GYU_SOULX_PYTHON"
printf '  GYU_SINGER_CACHE=%s\n' "$GYU_SINGER_CACHE"
printf '  GYU_V09_PACKAGE_DIR=%s\n' "$GYU_V09_PACKAGE_DIR"
printf '  GYU_SMOKE_OUTPUT_DIR=%s\n' "$GYU_SMOKE_OUTPUT_DIR"
printf '  GYU_SMOKE_PORT=%s\n' "$GYU_SMOKE_PORT"
echo ""

"$SCRIPT_DIR/openutau_v09_collect_approval_record.sh"
echo ""
echo "[ONECLICK] operational report:"
echo "$GYU_V09_READINESS_OUTPUT_DIR/operational_approval_record.md"
