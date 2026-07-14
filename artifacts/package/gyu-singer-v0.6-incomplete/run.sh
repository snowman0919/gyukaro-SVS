#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${GYU_SINGER_CACHE:?set GYU_SINGER_CACHE to pinned model cache}"
: "${GYU_SOULX_PYTHON:=$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python}"
export GYU_SINGER_CACHE GYU_SOULX_PYTHON
PYTHONPATH=src python -m gyu_singer.cli --backend gyu-singer-v0.6 --reference data/processed/master/216.wav render examples/quality_ko.json --output "${1:-output.wav}"
