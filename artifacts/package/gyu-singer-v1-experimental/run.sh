#!/bin/sh
PYTHONPATH=runtime .venv/bin/python -m gyu_singer.cli --backend neural --model model/gyu_moss_nano_sft --audio-tokenizer model/moss-audio-tokenizer-nano --reference model/gyu_reference.m4a render examples/smoke.json --output output.wav
