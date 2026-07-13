#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHONPATH=runtime .venv/bin/python -m gyu_singer.cli --backend hybrid-svs --checkpoint model/gyu_hybrid_v0.2.pt --audio-tokenizer model/moss-audio-tokenizer-nano --reference model/gyu_reference_216.wav render examples/smoke.json --output output.wav
