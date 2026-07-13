#!/bin/sh
set -eu
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
