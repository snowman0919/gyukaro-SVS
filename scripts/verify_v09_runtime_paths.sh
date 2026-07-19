#!/usr/bin/env sh
set -eu
ROOT=${1:-/tmp/gyu-singer-v0.9-openutau-test3}
RUNTIME_DIR=${GYU_SOULX_RUNTIME_DIR:-/home/kotori9/code/gyukaro/.venv-soulx}
CACHE_DIR=${GYU_SINGER_CACHE:-/home/kotori9/code/gyukaro/data/cache}
cd "$ROOT"
rm -f /tmp/v09-verify-serve.log /tmp/v09-verify-render.wav /tmp/v09-verify-smoke.log

export GYU_SOULX_RUNTIME_DIR="$RUNTIME_DIR"
export GYU_SINGER_CACHE="$CACHE_DIR"
unset GYU_SOULX_PYTHON

if [ -f "gyu-singer-v0.9-openutau/scripts/openutau_v09_runtime_smoke.sh" ]; then
  cd gyu-singer-v0.9-openutau
elif [ -f "scripts/openutau_v09_runtime_smoke.sh" ]; then
  :
else
  echo "cannot locate smoke script" >&2
  exit 2
fi

bash scripts/openutau_v09_runtime_smoke.sh > /tmp/v09-verify-smoke.log 2>&1
status=$?
if [ "$status" -ne 0 ]; then
  echo "smoke failed ($status)" >&2
  tail -n 40 /tmp/v09-verify-smoke.log >&2
  exit 2
fi

bash ./serve.sh 8777 >/tmp/v09-verify-serve.log 2>&1 &
SERVE_PID=$!
sleep 10
python - <<PY
import time, urllib.request
ok=False
for _ in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8777/health', timeout=2) as r:
            _=r.read(64)
        ok=True
        break
    except Exception:
        time.sleep(1)
if not ok:
    raise SystemExit('serve/health check failed')
PY

./render.sh examples/quality_ko.json /tmp/v09-verify-render.wav
if [ ! -s /tmp/v09-verify-render.wav ]; then
  echo "render produced empty file" >&2
  exit 2
fi
kill "$SERVE_PID" || true
wait "$SERVE_PID" 2>/dev/null || true
printf 'smoke_status=%s\nrender_size=%s\n' "$status" "$(stat -c%s /tmp/v09-verify-render.wav)"
