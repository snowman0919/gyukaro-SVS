#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
PACKAGE_ARG_WAS_SET=0
if [ -n "${GYU_V09_PACKAGE_DIR:-}" ]; then
  PACKAGE_ARG="$GYU_V09_PACKAGE_DIR"
  PACKAGE_ARG_WAS_SET=1
  PACKAGE_ZIP_HINT=""
  case "$PACKAGE_ARG" in
    *.zip)
      if [ -f "$PACKAGE_ARG" ]; then
        PACKAGE_ZIP_HINT="$PACKAGE_ARG"
        PACKAGE_DIR="${PACKAGE_ARG%.zip}"
      else
        PACKAGE_DIR="$PACKAGE_ARG"
      fi
      ;;
    *)
      PACKAGE_DIR="$PACKAGE_ARG"
      ;;
  esac
fi
OUTPUT_DIR="${GYU_V09_READINESS_OUTPUT_DIR:-$ROOT/artifacts/reports/openutau_v09}"

if [ ! -d "$PACKAGE_DIR" ] && [ -f "$ROOT/serve.sh" ] && [ -f "$ROOT/scripts/openutau_v09_runtime_smoke.sh" ]; then
  if [ "$PACKAGE_ARG_WAS_SET" -eq 0 ]; then
    PACKAGE_DIR="$ROOT"
  fi
fi

if [ ! -d "$PACKAGE_DIR" ] && [ -z "${PACKAGE_ZIP_HINT:-}" ]; then
  echo "cannot find package dir: $PACKAGE_DIR" >&2
  exit 2
fi

if [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/.venv-soulx/.venv/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../../.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../../.venv-soulx"
elif [ -z "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -x "$ROOT/../../../.venv-soulx/bin/python" ]; then
  GYU_SOULX_RUNTIME_DIR="$ROOT/../../../.venv-soulx"
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
  elif [ -x "$ROOT/../.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/../../.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/../../../.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../../.venv-soulx/.venv/bin/python"
  elif [ -x "$HOME/.venv-soulx/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/.venv/bin/python"
  elif [ -x "$ROOT/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/.venv-soulx/bin/python"
  elif [ -x "$ROOT/../.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../.venv-soulx/bin/python"
  elif [ -x "$ROOT/../../.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../.venv-soulx/bin/python"
  elif [ -x "$ROOT/../../../.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$ROOT/../../../.venv-soulx/bin/python"
  elif [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  fi
fi

if [ -n "${GYU_SOULX_PYTHON:-}" ]; then
  export GYU_SOULX_PYTHON
fi
if [ -n "${GYU_SOULX_RUNTIME_DIR:-}" ]; then
  export GYU_SOULX_RUNTIME_DIR
fi

if [ -z "${GYU_SINGER_CACHE:-}" ]; then
  if [ -d "$ROOT/data/cache" ]; then
    GYU_SINGER_CACHE="$ROOT/data/cache"
  elif [ -d "$ROOT/../../data/cache" ]; then
    GYU_SINGER_CACHE="$ROOT/../../data/cache"
  fi
fi
export GYU_SINGER_CACHE

ensure_package_dir() {
  if [ -d "$PACKAGE_DIR" ]; then
    return 0
  fi

  package_zip="${PACKAGE_ZIP_HINT:-${PACKAGE_DIR}.zip}"
  if [ ! -f "$package_zip" ]; then
    package_zip="$ROOT/artifacts/package/$(basename "$PACKAGE_DIR").zip"
  fi

  if [ ! -f "$package_zip" ]; then
    echo "cannot find package dir or archive: $PACKAGE_DIR (or $package_zip)" >&2
    return 2
  fi

  package_parent="$(dirname "$PACKAGE_DIR")"
  tmp_extract_dir="$(mktemp -d "${package_parent}/.gyu-v09-pkg.XXXXXX")"
  mkdir -p "$tmp_extract_dir"
  unzip -q "$package_zip" -d "$tmp_extract_dir"
  extracted_dirs_count="$(find "$tmp_extract_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  if [ "$extracted_dirs_count" -eq 0 ]; then
    rm -rf "$tmp_extract_dir"
    echo "package archive contains no top-level directory: $package_zip" >&2
    return 2
  fi
  if [ "$extracted_dirs_count" -eq 1 ]; then
    extracted_dir="$(find "$tmp_extract_dir" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  else
    extracted_dir="$(find "$tmp_extract_dir" -mindepth 1 -maxdepth 1 -type d | sort | awk 'NR==1 {print; exit}')"
  fi
  if [ -z "${extracted_dir:-}" ]; then
    rm -rf "$tmp_extract_dir"
    echo "package archive extracted but target dir missing: $PACKAGE_DIR" >&2
    return 2
  fi
  rm -rf "$PACKAGE_DIR"
  mkdir -p "$package_parent"
  if [ "$extracted_dir" != "$PACKAGE_DIR" ]; then
    mv "$extracted_dir" "$PACKAGE_DIR"
  fi
  rm -rf "$tmp_extract_dir"
  if [ ! -d "$PACKAGE_DIR" ]; then
    echo "package archive extracted but target dir missing: $PACKAGE_DIR" >&2
    return 2
  fi
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
