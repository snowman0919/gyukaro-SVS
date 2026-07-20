#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export GYU_V09_PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau}"
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-/home/kotori9/code/gyukaro/data/cache}"
export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-/home/kotori9/code/gyukaro/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-/home/kotori9/code/gyukaro/.venv-soulx/bin/python}"

MODE="start"
PORT="8765"
arg1="${1-}"
arg2="${2-}"
case "$arg1" in
  -h) arg1="--help" ;;
esac
case "$arg2" in
  -h) arg2="--help" ;;
esac

if [ -n "$arg1" ] && [ "${arg1#--}" != "$arg1" ]; then
  MODE="${arg1#--}"
  case "$MODE" in
    status|stop|run)
      if [ -n "${arg2}" ] && [ "${arg2#--}" = "$arg2" ]; then
        PORT="$arg2"
      elif [ "$arg2" = "--help" ] || [ "$arg2" = "-h" ]; then
        MODE="help"
      fi
      ;;
    help|h)
      MODE="help"
      ;;
    *)
      echo "unknown option: $arg1" >&2
      exit 2
      ;;
  esac
else
  if [ -n "$arg1" ]; then
    PORT="$arg1"
  fi
  if [ -n "$arg2" ]; then
    case "${arg2#--}" in
      status|stop|run)
        MODE="${arg2#--}"
        ;;
      help|h)
        MODE="help"
        ;;
      *)
        echo "unknown option: $arg2" >&2
        exit 2
        ;;
    esac
  fi
fi
PID_FILE="${GYU_V09_SERVICE_PID_FILE:-/tmp/gyu-v09-go-live.pid}"
LOG_FILE="${GYU_V09_SERVICE_LOG:-/tmp/gyu-v09-go-live.log}"

case "$MODE" in
  stop)
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
      echo "invalid PORT: $PORT (expected integer 1..65535)" >&2
      exit 2
    fi
    if [ -f "$PID_FILE" ]; then
      pid="$(cat "$PID_FILE")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "stopped pid=$pid"
      fi
      rm -f "$PID_FILE"
    else
      echo "no pid file"
    fi
    exit 0
    ;;
  status)
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
      echo "invalid PORT: $PORT (expected integer 1..65535)" >&2
      exit 2
    fi
    if health="$(curl -sSfS "http://127.0.0.1:${PORT}/health")"; then
      printf 'service alive on %s\n' "$PORT"
      printf '%s\n' "$health"
      exit 0
    fi
    exit 2
    ;;
  help|-h)
  cat <<EOF
Usage: $(basename "$0") [PORT] [--stop|--status|--run|--help]
       $(basename "$0") --status [PORT]
       $(basename "$0") --stop [PORT]
       $(basename "$0") -h | --help
  modes:
    --run: run resident serve in foreground (for systemd unit)
  default PORT=8765
EOF
    exit 0
    ;;
esac

validate_env() {
  if [ -n "${GYU_SOULX_PYTHON}" ] && [ ! -x "${GYU_SOULX_PYTHON}" ]; then
    echo "missing GYU_SOULX_PYTHON: ${GYU_SOULX_PYTHON}" >&2
    exit 2
  fi
  if [ ! -d "$GYU_SINGER_CACHE" ]; then
    echo "missing GYU_SINGER_CACHE: ${GYU_SINGER_CACHE}" >&2
    exit 2
  fi
}

run_service() {
  validate_env
  echo "[go-live] foreground service: port=$PORT cache=$GYU_SINGER_CACHE"
  exec ./serve.sh "$PORT"
}

if [ "$MODE" = "run" ]; then
  run_service
fi

validate_env

check_port_free() {
  if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "invalid PORT: $PORT (expected integer 1..65535)" >&2
    exit 2
  fi
  local pids
  pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
  if [ -z "$pids" ]; then
    return
  fi
  if [ "${GYU_V09_AUTO_KILL_STALE_PORT:-0}" = "1" ]; then
    if command -v kill >/dev/null 2>&1; then
      printf 'port %s is already in use (pids: %s); auto-clean enabled, attempting terminate.\n' "$PORT" "$pids" >&2
      printf '%s\n' "$pids" | xargs -r kill -9
      sleep 1
      return
    fi
  fi
  printf 'port %s is already in use by pid(s): %s\n' "$PORT" "$pids" >&2
  printf 'If this is stale/leftover gyu_singer process, either stop it first or set GYU_V09_AUTO_KILL_STALE_PORT=1\n' >&2
  exit 4
}

if [ -f "$PID_FILE" ]; then
  old="$(cat "$PID_FILE")"
  if kill -0 "$old" 2>/dev/null; then
    echo "existing service running pid=${old}; stop first with: $0 $PORT --stop"
    exit 3
  fi
  rm -f "$PID_FILE"
fi

check_port_free

echo "[go-live] booting service: port=$PORT cache=$GYU_SINGER_CACHE"
nohup ./serve.sh "$PORT" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
NEW_PID="$!"
boot_ok=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  sleep 0.5
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "failed to start service process (pid $NEW_PID); log: $LOG_FILE" >&2
    cat "$LOG_FILE" >&2 || true
    exit 2
  fi
  LISTEN_PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
  if [ -n "$LISTEN_PIDS" ] && printf '%s\n' "$LISTEN_PIDS" | grep -qx "$NEW_PID"; then
    boot_ok=1
    break
  fi
done
if [ "$boot_ok" -ne 1 ]; then
  if [ -z "$LISTEN_PIDS" ]; then
    echo "no process is listening on port $PORT after start; log: $LOG_FILE" >&2
  else
    echo "started pid $NEW_PID is not the listener for port $PORT; port occupied by: $LISTEN_PIDS; log: $LOG_FILE" >&2
  fi
  cat "$LOG_FILE" >&2 || true
  exit 2
fi

if ! curl -sSfS "http://127.0.0.1:${PORT}/health" >/tmp/gyu-v09-go-live-health.json; then
  echo "health check failed. log: $LOG_FILE" >&2
  exit 2
fi

cat >/tmp/gyu-v09-go-live-check.json <<JSON
{"status":"ok","pid":$(cat "$PID_FILE"),"service_url":"http://127.0.0.1:${PORT}","health_file":"/tmp/gyu-v09-go-live-health.json","log":"$LOG_FILE"}
JSON
echo "[go-live] up. pid=$(cat "$PID_FILE")"
echo "[go-live] health=OK"
echo "[go-live] service_url=http://127.0.0.1:${PORT}"
echo "[go-live] tail: tail -f $LOG_FILE"
