#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME=$ROOT/.runtime
[ -f "$RUNTIME/dotnet-path" ] || { echo "run ./install.sh first" >&2; exit 2; }
DOTNET=$(cat "$RUNTIME/dotnet-path")
export GYU_SINGER_CACHE=$RUNTIME/cache
export GYU_SOULX_PYTHON=$RUNTIME/soulx-venv/bin/python
export GYU_RENDERER_URL=http://127.0.0.1:8765/render
cd "$ROOT"
"$ROOT/serve.sh" 8765 >"$RUNTIME/resident.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true' EXIT INT TERM
"$GYU_SOULX_PYTHON" -c 'import json,time,urllib.request
for _ in range(240):
 try:
  assert json.load(urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2))["status"] == "ok"; break
 except Exception: time.sleep(.25)
else: raise SystemExit("resident health timeout")'
"$DOTNET" "$RUNTIME/OpenUtau/OpenUtau/bin/Release/net8.0/OpenUtau.dll" "$@"
