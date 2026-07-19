#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

OUTPUT_MD="${OPENUTAU_V09_APPROVAL_REPORT:-$ROOT/artifacts/reports/openutau_v09/operational_approval_record.md}"
OP_OUTPUT_DIR="${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}"
PKG_DIR="${GYU_V09_PACKAGE_DIR:-$ROOT/artifacts/package/gyu-singer-v0.9-openutau}"

echo "[APPROVAL] run production readiness check"
"$SCRIPT_DIR/openutau_v09_production_readiness.sh"

READY_SUMMARY="$ROOT/artifacts/reports/openutau_v09/readiness_summary.json"
BEHAVIOR_JSON="$OP_OUTPUT_DIR/openutau_v09_operational_behavior.json"
SMOKE_WAV="$OP_OUTPUT_DIR/openutau_v09_smoke.wav"
PKG_REPORT="$ROOT/artifacts/reports/openutau_v09/behavior.json"

for path in "$READY_SUMMARY" "$BEHAVIOR_JSON" "$SMOKE_WAV"; do
  if [ ! -f "$path" ]; then
    echo "missing $path" >&2
    exit 2
  fi

done

mkdir -p "$(dirname "$OUTPUT_MD")"

python - "$READY_SUMMARY" "$BEHAVIOR_JSON" "$SMOKE_WAV" "$OUTPUT_MD" "$PKG_DIR" "$PKG_REPORT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ready_path = Path(sys.argv[1])
behavior_path = Path(sys.argv[2])
smoke_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
pkg_dir = Path(sys.argv[5])
pkg_report = Path(sys.argv[6])

ready = json.loads(ready_path.read_text(encoding="utf-8"))
behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
smoke_sha = hashlib.sha256(smoke_path.read_bytes()).hexdigest()

gates = behavior.get("gates", {})
gate_lines = [f"- {k}: {str(v).lower()}" for k, v in sorted(gates.items())]

lines = []
lines.append("# OpenUtau v0.9 실사용 운영 승인 기록 (Runtime-path 기준)")
lines.append("")
lines.append(f"생성시각: {datetime.now(timezone.utc).isoformat()}")
lines.append(f"패키지: `{pkg_dir}`")
lines.append(f"readiness 요약: `{ready_path}`")
lines.append(f"behavior JSON: `{behavior_path}`")
lines.append(f"smoke WAV: `{smoke_path}`")
lines.append("")

lines.append("## 1) 통과 요약")
lines.append(f"- READY: `{str(ready.get('READY', False)).lower()}`")
dataset_ok = ready.get("dataset_validation", {}).get("pass")
lines.append(f"- dataset validation: `{str(dataset_ok).lower()}`")
pytest_ok = ready.get("pytest_check", {}).get("pass")
lines.append(f"- pytest: `{str(pytest_ok).lower()}`")
smoke_pass = ready.get("verify_v09_runtime_paths", {}).get("smoke_status_0")
lines.append(f"- verify_v09_runtime_paths: `{str(smoke_pass).lower()}`")
operational_ok = ready.get("operational_check", {}).get("pass")
lines.append(f"- operational check: `{str(operational_ok).lower()}`")
package_hash_match = ready.get("package", {}).get("hash_match")
if package_hash_match is not None:
    declared = ready.get("package", {}).get("declared_sha256", "-")
    lines.append(f"- 패키지 해시 일치: `{str(package_hash_match).lower()}` (`{declared}`)")
else:
    lines.append("- 패키지 해시: 확인 실패")
lines.append(f"- smoke 파일 SHA-256: `{smoke_sha}`")
lines.append(f"- smoke size: `{smoke_path.stat().st_size}` bytes")
lines.append("")

lines.append("## 2) 형식 검증")
formats = behavior.get("formats") or {}
if formats:
    for language, info in sorted(formats.items()):
        sr = info.get("sample_rate", "-")
        ch = info.get("channels", "-")
        sec = info.get("seconds", "-")
        lines.append(f"- {language}: sample_rate={sr}, channels={ch}, seconds={sec}")
else:
    lines.append("- format 정보가 behavior JSON에 없음")

lines.append("")
lines.append("## 3) behavioral gates")
if gate_lines:
    lines.extend(gate_lines)
else:
    lines.append("- gates 정보가 behavior JSON에 없음")

lines.append("")
lines.append("## 4) 경로 고정 실행 커맨드")
lines.append("```sh")
lines.append("export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx")
lines.append("export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python")
lines.append("export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache")
lines.append(f'export GYU_V09_PACKAGE_DIR="{pkg_dir}"')
lines.append("cd /home/kotori9/code/gyukaro")
lines.append("./scripts/openutau_v09_production_readiness.sh")
lines.append("```")
lines.append("")

lines.append("## 5) 판단")
lines.append("지정된 runtime 경로 기준으로 오퍼레이션 체크 통과." if ready.get("READY") else "오퍼레이션 체크 미통과. 위 항목 확인 필요.")

if pkg_report.exists():
    lines.append("")
    lines.append("## 6) 최신 패키지 내역")
    lines.append(f"- 패키지 보고서: `{pkg_report}`")

output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "written: $OUTPUT_MD"
