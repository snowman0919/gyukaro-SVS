#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME=$ROOT/.runtime
export GYU_SINGER_CACHE=$RUNTIME/cache
export GYU_SOULX_PYTHON=$RUNTIME/soulx-venv/bin/python
export TORCH_HOME=$RUNTIME/cache/torch
cd "$ROOT"
exec "$GYU_SOULX_PYTHON" -m gyu_singer.cli --backend gyu-singer-rc5 --reference model/gyu_reference_216.wav serve --port "${1:-8765}"
