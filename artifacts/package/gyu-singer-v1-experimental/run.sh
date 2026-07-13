#!/bin/sh
PYTHONPATH=runtime python -m gyu_singer.cli --model model/gyu_v1_experimental.npz render examples/korean.json --output output.wav
