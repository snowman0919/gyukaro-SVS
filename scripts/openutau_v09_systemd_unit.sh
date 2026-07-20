#!/usr/bin/env bash
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PORT="${1:-8765}"
UNIT_NAME="${2:-gyu-openutau-v09.service}"
ACTION="${3:-print}"

action="${ACTION#--}"

emit_unit() {
cat <<EOF
[Unit]
Description=GYU OpenUtau v0.9 backend
After=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=3
Environment=GYU_V09_PACKAGE_DIR=$ROOT/artifacts/package/gyu-singer-v0.9-openutau
Environment=GYU_SINGER_CACHE=$ROOT/data/cache
Environment=GYU_SOULX_RUNTIME_DIR=$ROOT/.venv-soulx
Environment=GYU_SOULX_PYTHON=$ROOT/.venv-soulx/bin/python
WorkingDirectory=$ROOT
ExecStart=$ROOT/scripts/openutau_v09_go_live.sh $PORT
ExecStop=$ROOT/scripts/openutau_v09_go_live.sh --stop $PORT
StandardOutput=append:$ROOT/artifacts/reports/openutau_v09/openutau-v09-service.log
StandardError=append:$ROOT/artifacts/reports/openutau_v09/openutau-v09-service.err

[Install]
WantedBy=default.target
EOF
}

case "$action" in
  print|show)
    emit_unit
    ;;
  apply)
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    mkdir -p "$UNIT_DIR"
    mkdir -p "$ROOT/artifacts/reports/openutau_v09"
    emit_unit > "$UNIT_DIR/$UNIT_NAME"
    echo "wrote $UNIT_DIR/$UNIT_NAME"
    systemctl --user daemon-reload
    echo "enable: systemctl --user enable $UNIT_NAME"
    echo "start : systemctl --user start $UNIT_NAME"
    ;;
  help|-h|--help)
    cat <<EOF
Usage: $(basename "$0") [PORT] [UNIT_NAME] [print|apply]
  default PORT=8765
  default UNIT_NAME=gyu-openutau-v09.service
  print: print unit content (default)
  apply: write to ~/.config/systemd/user and reload user daemon
EOF
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac
