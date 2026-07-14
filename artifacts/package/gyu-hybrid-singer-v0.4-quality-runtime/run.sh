#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${GYU_SINGER_CACHE:?set cache path, or run sh bootstrap.sh /path/to/cache}"
: "${GYU_SOULX_PYTHON:=$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python}"
export GYU_SINGER_CACHE GYU_SOULX_PYTHON
PYTHONPATH=runtime python -m gyu_singer.cli --backend hybrid-svs --reference model/gyu_reference_216.wav render examples/quality_ko.json --output output.wav
