#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME=$ROOT/.runtime
CACHE_SOURCE=
SKIP_RENDER=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cache-source) CACHE_SOURCE=${2:?--cache-source needs a path}; shift 2 ;;
    --skip-render-smoke) SKIP_RENDER=1; shift ;;
    *) echo "usage: ./install.sh [--cache-source /existing/pinned/cache] [--skip-render-smoke]" >&2; exit 2 ;;
  esac
done

[ "$(uname -s)" = Linux ] || { echo "v1.0 supports Linux only" >&2; exit 2; }
case "$(uname -m)" in x86_64|aarch64) ;; *) echo "unsupported architecture: $(uname -m)" >&2; exit 2 ;; esac
for command in python3 git; do command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }; done
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
mkdir -p "$RUNTIME"

PY=$RUNTIME/soulx-venv/bin/python
if [ ! -x "$PY" ]; then python3 -m venv --system-site-packages "$RUNTIME/soulx-venv"; fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$ROOT/soulx-runtime-requirements.txt"
"$PY" -m pip install -q --no-deps "$ROOT"

if [ -n "$CACHE_SOURCE" ]; then
  "$PY" "$ROOT/model-downloader.py" --runtime "$RUNTIME" --manifest "$ROOT/model-dependencies.json" --cache-source "$CACHE_SOURCE"
else
  "$PY" "$ROOT/model-downloader.py" --runtime "$RUNTIME" --manifest "$ROOT/model-dependencies.json"
fi

OMNI_PY=$RUNTIME/cache/omnivoice/.venv/bin/python
if [ ! -x "$OMNI_PY" ]; then python3 -m venv "$RUNTIME/cache/omnivoice/.venv"; fi
"$OMNI_PY" -m pip install -q --upgrade pip
"$OMNI_PY" -m pip install -q -e "$RUNTIME/cache/omnivoice"

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print("CUDA:", torch.cuda.get_device_name())'
"$PY" "$ROOT/model-downloader.py" --runtime "$RUNTIME" --manifest "$ROOT/model-dependencies.json" --verify-only

if [ -n "${GYU_DOTNET:-}" ]; then
  DOTNET=$GYU_DOTNET
elif command -v dotnet >/dev/null 2>&1; then
  DOTNET=$(command -v dotnet)
else
  command -v curl >/dev/null || { echo "curl is required to install local .NET 8" >&2; exit 2; }
  mkdir -p "$RUNTIME/dotnet"
  curl -fsSL https://dot.net/v1/dotnet-install.sh -o "$RUNTIME/dotnet-install.sh"
  sh "$RUNTIME/dotnet-install.sh" --channel 8.0 --install-dir "$RUNTIME/dotnet"
  DOTNET=$RUNTIME/dotnet/dotnet
fi
"$DOTNET" --version
printf '%s\n' "$DOTNET" > "$RUNTIME/dotnet-path"

OPENUTAU=$RUNTIME/OpenUtau
OPENUTAU_REV=27573ac5c888d927119d5f65a207312d79194b1f
if [ ! -d "$OPENUTAU/.git" ]; then git clone https://github.com/stakira/OpenUtau.git "$OPENUTAU"; fi
git -C "$OPENUTAU" checkout --detach "$OPENUTAU_REV"
if [ ! -f "$OPENUTAU/.gyu-overlay-$OPENUTAU_REV" ]; then
  "$ROOT/integrations/openutau/install_fork.sh" "$OPENUTAU"
  : > "$OPENUTAU/.gyu-overlay-$OPENUTAU_REV"
fi
"$DOTNET" build "$OPENUTAU/OpenUtau.csproj" -c Release

chmod +x "$ROOT/serve.sh" "$ROOT/render-example.sh" "$ROOT/launch-openutau.sh"
if [ "$SKIP_RENDER" -eq 0 ]; then
  "$ROOT/render-example.sh" "$ROOT/examples/quality_ko.json" "$RUNTIME/install-smoke.wav"
  "$PY" "$ROOT/verify-install.py" --root "$ROOT" --audio "$RUNTIME/install-smoke.wav"
else
  "$PY" "$ROOT/verify-install.py" --root "$ROOT"
fi
echo "Install complete. Run: $ROOT/launch-openutau.sh"
