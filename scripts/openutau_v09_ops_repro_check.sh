#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT"

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$ROOT/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-$ROOT/.venv-soulx/bin/python}"
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-$ROOT/data/cache}"
export GYU_V09_PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
export GYU_SMOKE_OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check-repro}"
export GYU_SMOKE_PORT="${GYU_SMOKE_PORT:-8780}"
export GYU_OPS_RUN_PKG_TESTS=1

report_step() {
  printf '\n[OPS-REPRO] %s\n' "$1"
}

report_step "validate_dataset"
python scripts/validate_dataset.py

report_step "ops_check (includes package tests)"
if ! scripts/openutau_v09_ops_check.sh; then
  echo "[OPS-REPRO] ops_check FAILED" >&2
  exit 2
fi

report_step "oneclick operational check"
if ! scripts/openutau_v09_oneclick_operational_check.sh; then
  echo "[OPS-REPRO] oneclick FAILED" >&2
  exit 2
fi

report_step "done"
echo "[OPS-REPRO] PASS"
echo "  readiness: $ROOT/artifacts/reports/openutau_v09/readiness_summary.json"
echo "  behavior:  ${GYU_SMOKE_OUTPUT_DIR}/openutau_v09_operational_behavior.json"
echo "  smoke:     ${GYU_SMOKE_OUTPUT_DIR}/openutau_v09_smoke.wav"
