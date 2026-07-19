#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
OUTPUT_DIR="${GYU_V09_READINESS_OUTPUT_DIR:-$ROOT/artifacts/reports/openutau_v09}"

ensure_package_dir() {
  if [ -d "$PACKAGE_DIR" ]; then
    return 0
  fi

  package_zip="${PACKAGE_DIR}.zip"
  if [ ! -f "$package_zip" ]; then
    package_zip="$ROOT/artifacts/package/$(basename "$PACKAGE_DIR").zip"
  fi

  if [ ! -f "$package_zip" ]; then
    echo "cannot find package dir or archive: $PACKAGE_DIR (or $package_zip)" >&2
    return 2
  fi

  rm -rf "$PACKAGE_DIR"
  unzip -q "$package_zip" -d "$ROOT/artifacts/package/"
}

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$ROOT/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-$ROOT/.venv-soulx/bin/python}"
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-$ROOT/data/cache}"
export GYU_SMOKE_OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}"

if ! ensure_package_dir; then
  exit 2
fi

python "$ROOT/scripts/report_openutau_v09_readiness.py" \
  --package "$PACKAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --runtime-dir "$GYU_SOULX_RUNTIME_DIR" \
  --python "$GYU_SOULX_PYTHON" \
  --cache "$GYU_SINGER_CACHE" \
  --operational-output-dir "$GYU_SMOKE_OUTPUT_DIR"
