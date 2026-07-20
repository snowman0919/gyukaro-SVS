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

if [ -f "$PID_FILE" ]; then
  old="$(cat "$PID_FILE")"
  if kill -0 "$old" 2>/dev/null; then
    echo "existing service running pid=${old}; stop first with: $0 $PORT --stop"
    exit 3
  fi
  rm -f "$PID_FILE"
fi

echo "[go-live] booting service: port=$PORT cache=$GYU_SINGER_CACHE"
nohup ./serve.sh "$PORT" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 2

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
