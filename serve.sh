#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

: "${GYU_SINGER_CACHE:?set GYU_SINGER_CACHE to the pinned model cache}"
if [ ! -d "$GYU_SINGER_CACHE" ]; then
  echo "missing GYU_SINGER_CACHE: $GYU_SINGER_CACHE"
  exit 2
fi

if [ -z "${GYU_SOULX_PYTHON:-}" ]; then
  if [ -x "$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python"
  elif [ -x "$GYU_SINGER_CACHE/soulx-singer/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SINGER_CACHE/soulx-singer/.venv-soulx/bin/python"
  elif [ -x "$SCRIPT_DIR/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$SCRIPT_DIR/.venv-soulx/bin/python"
  elif [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  fi
fi

: "${GYU_SOULX_PYTHON:?set GYU_SOULX_PYTHON to the pinned SoulX Python}"
if [ ! -x "$GYU_SOULX_PYTHON" ]; then
  echo "invalid GYU_SOULX_PYTHON: $GYU_SOULX_PYTHON"
  exit 2
fi

if [ ! -x "$GYU_SINGER_CACHE/omnivoice/.venv/bin/python" ]; then
  echo "missing pinned OmniVoice runtime: $GYU_SINGER_CACHE/omnivoice/.venv/bin/python"
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,expandable_segments:True}"
export GYU_SINGER_CACHE GYU_SOULX_PYTHON

exec env PYTHONPATH=src "${GYU_SOULX_PYTHON}" -m gyu_singer.cli --backend gyu-singer-v0.8 --reference data/processed/master/216.wav serve --port "${1:-8765}"
