#!/bin/sh
set -eu
CACHE=${1:?usage: sh bootstrap.sh /path/to/cache}
mkdir -p "$CACHE"
if [ ! -d "$CACHE/omnivoice/.git" ]; then git clone https://github.com/k2-fsa/OmniVoice.git "$CACHE/omnivoice"; fi
git -C "$CACHE/omnivoice" checkout 1574e06
python3 -m venv "$CACHE/omnivoice/.venv"
"$CACHE/omnivoice/.venv/bin/pip" install -e "$CACHE/omnivoice"
"$CACHE/omnivoice/.venv/bin/hf" download ModelsLab/omnivoice-singing --local-dir "$CACHE/omnivoice-checkpoint"
if [ ! -d "$CACHE/soulx-singer/.git" ]; then git clone https://github.com/Soul-AILab/SoulX-Singer.git "$CACHE/soulx-singer"; fi
git -C "$CACHE/soulx-singer" checkout 81aeb3a
python3 -m venv "$CACHE/soulx-singer/.venv"
"$CACHE/soulx-singer/.venv/bin/pip" install -r "$CACHE/soulx-singer/requirements.txt"
"$CACHE/soulx-singer/.venv/bin/pip" install -U huggingface_hub
"$CACHE/soulx-singer/.venv/bin/hf" download Soul-AILab/SoulX-Singer --local-dir "$CACHE/soulx-singer/pretrained_models/SoulX-Singer"
"$CACHE/soulx-singer/.venv/bin/hf" download Soul-AILab/SoulX-Singer-Preprocess --local-dir "$CACHE/soulx-singer/pretrained_models/SoulX-Singer-Preprocess"
printf '%s\n' "export GYU_SINGER_CACHE=$CACHE" "export GYU_SOULX_PYTHON=$CACHE/soulx-singer/.venv/bin/python"
