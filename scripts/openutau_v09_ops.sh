#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$ROOT/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-$ROOT/.venv-soulx/bin/python}"
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-$ROOT/data/cache}"
export GYU_V09_PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
export GYU_SMOKE_PORT="${GYU_SMOKE_PORT:-8780}"

SUBCMD="${1:-}"
case "$SUBCMD" in
  start)
    cd "$ROOT"
    "$SCRIPT_DIR/openutau_v09_go_live.sh" "$GYU_SMOKE_PORT"
    ;;
  stop)
    "$SCRIPT_DIR/openutau_v09_go_live.sh" --stop "$GYU_SMOKE_PORT"
    ;;
  status)
    "$SCRIPT_DIR/openutau_v09_go_live.sh" --status "$GYU_SMOKE_PORT"
    ;;
  start-systemd)
    "$SCRIPT_DIR/openutau_v09_systemd_unit.sh" "$GYU_SMOKE_PORT" gyu-openutau-v0.9.service apply
    systemctl --user daemon-reload
    systemctl --user enable gyu-openutau-v0.9.service
    systemctl --user start gyu-openutau-v0.9.service
    ;;
  stop-systemd)
    systemctl --user stop gyu-openutau-v0.9.service || true
    systemctl --user disable gyu-openutau-v0.9.service || true
    ;;
  systemd-status)
    systemctl --user status gyu-openutau-v0.9.service --no-pager -l
    ;;
  readiness)
    "$SCRIPT_DIR/openutau_v09_production_readiness.sh"
    ;;
  *)
    cat <<EOF
Usage: $(basename "$0") {start|stop|status|start-systemd|stop-systemd|systemd-status|readiness}
  env: GYU_SOULX_RUNTIME_DIR GYU_SOULX_PYTHON GYU_SINGER_CACHE GYU_SMOKE_PORT
EOF
    exit 2
    ;;
esac
