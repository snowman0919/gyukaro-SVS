#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME=$ROOT/.runtime
export GYU_SINGER_CACHE=$RUNTIME/cache
export GYU_SOULX_PYTHON=$RUNTIME/soulx-venv/bin/python
cd "$ROOT"
exec "$GYU_SOULX_PYTHON" -m gyu_singer.cli --backend gyu-singer-v0.8 --reference model/gyu_reference_216.wav render "${1:-examples/quality_ko.json}" --output "${2:-output.wav}"
