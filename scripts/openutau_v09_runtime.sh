#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

export GYU_SOULX_RUNTIME_DIR="${GYU_SOULX_RUNTIME_DIR:-$ROOT/.venv-soulx}"
export GYU_SOULX_PYTHON="${GYU_SOULX_PYTHON:-$ROOT/.venv-soulx/bin/python}"
export GYU_SINGER_CACHE="${GYU_SINGER_CACHE:-$ROOT/data/cache}"
export GYU_V09_PACKAGE_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"
export GYU_V09_READINESS_OUTPUT_DIR="${GYU_V09_READINESS_OUTPUT_DIR:-$ROOT/artifacts/reports/openutau_v09}"
export GYU_SMOKE_OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}"
export GYU_SMOKE_PORT="${GYU_SMOKE_PORT:-8780}"

SUBCMD="${1:-start}"

case "$SUBCMD" in
  -h|--help|help)
    cat <<'EOF_USAGE'
Usage: scripts/openutau_v09_runtime.sh {start|stop|status|readiness|ops-check|ops-full|systemd-start|systemd-stop|systemd-status|systemd-restart|systemd-quickstart}
  start        Start v0.9 service (default)
  stop         Stop v0.9 service
  status       Check health on GYU_SMOKE_PORT
  readiness    Run production readiness check (dataset + scripted + operational)
  ops-check    Run ops check (dataset + behavior + optional package tests)
  ops-full     Run ops check with package smoke tests (set GYU_OPS_RUN_PKG_TESTS=1)
  systemd-start|systemd-stop|systemd-status|systemd-restart|systemd-quickstart

Environment variables default to pinned local paths:
  GYU_SOULX_RUNTIME_DIR=$ROOT/.venv-soulx
  GYU_SOULX_PYTHON=$ROOT/.venv-soulx/bin/python
  GYU_SINGER_CACHE=$ROOT/data/cache
  GYU_V09_PACKAGE_DIR=$ROOT/artifacts/package/gyu-singer-v0.9-openutau
EOF_USAGE
    exit 0
    ;;
  start)
    "$SCRIPT_DIR/openutau_v09_ops.sh" start
    ;;
  stop)
    "$SCRIPT_DIR/openutau_v09_ops.sh" stop
    ;;
  status)
    "$SCRIPT_DIR/openutau_v09_ops.sh" status
    ;;
  readiness)
    "$SCRIPT_DIR/openutau_v09_production_readiness.sh"
    ;;
  ops-check)
    "$SCRIPT_DIR/openutau_v09_ops_check.sh"
    ;;
  ops-full)
    GYU_OPS_RUN_PKG_TESTS=1 "$SCRIPT_DIR/openutau_v09_ops_check.sh"
    ;;
  systemd-start)
    "$SCRIPT_DIR/openutau_v09_ops.sh" start-systemd
    ;;
  systemd-stop)
    "$SCRIPT_DIR/openutau_v09_ops.sh" stop-systemd
    ;;
  systemd-status)
    "$SCRIPT_DIR/openutau_v09_ops.sh" systemd-status
    ;;
  systemd-restart)
    "$SCRIPT_DIR/openutau_v09_ops.sh" systemd-restart
    ;;
  systemd-quickstart)
    "$SCRIPT_DIR/openutau_v09_ops.sh" systemd-quickstart
    ;;
  *)
    echo "unknown command: $SUBCMD" >&2
    echo "run: scripts/openutau_v09_runtime.sh -h" >&2
    exit 2
    ;;
esac
