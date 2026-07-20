# OpenUtau v0.9 런타임 Readiness 1-Pager

- 작성시각: 2026-07-20T22:43:50.501913Z
- 대상: 지정 런타임 경로 기준 실사용 실행성 검증

## 결론
- 상태: **실행 가능(현재 기준, PASS)**
- openutau 사용 전제: `openutau_v09_collect_approval_record.sh` / `openutau_v09_ops_repro_check.sh` 경로 고정 실행)

## 핵심 게이트
- READY: True
- dataset validation: True
- pytest: True
- runtime path check: True
- operational behavior: True
- package hash match: True (d153383e10610afdbca9b2ff4624582b145e1f35d33bc1eace399ea48fcb0a10)
- behavioral gates pass: True

## 증적
- readiness: artifacts/reports/openutau_v09/readiness_summary.json
- behavior: /tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json
- smoke: /tmp/gyu-v09-operational-check/openutau_v09_smoke.wav
- smoke_sha256: 8cc08abadd0f5fc381ae9caea15fcae04072954ff489a8ca254230a1100c04eb
- approval: artifacts/reports/openutau_v09/operational_approval_record.md

## 출력 형식(요약)
- en: SR=48000, ch=1, sec=2.12
- ja: SR=48000, ch=1, sec=2.12
- ko: SR=48000, ch=1, sec=2.12
- lyric_edit: SR=48000, ch=1, sec=2.12
- note_pitch_edit: SR=48000, ch=1, sec=2.12
- style_energetic: SR=48000, ch=1, sec=2.12
- user_pitch_edit: SR=48000, ch=1, sec=2.12

## 고정 실행 커맨드(재현)
```sh
cd /home/kotori9/code/gyukaro
export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx
export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python
export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache
export GYU_V09_PACKAGE_DIR=/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau
export GYU_SMOKE_OUTPUT_DIR=/tmp/gyu-v09-operational-check
export GYU_SMOKE_PORT=8780
export GYU_OPS_RUN_PKG_TESTS=1
./scripts/openutau_v09_collect_approval_record.sh
```

## 재현 체크
1) `./scripts/openutau_v09_ops_repro_check.sh` (재현용)
2) `artifacts/reports/openutau_v09/operational_approval_record.md` 최신화 확인
3) readiness/gate/behavior 파일 존재 확인

## 제한/주의
- 실사용 배포 승인과 동일한 것은 아님; 현재는 runtime readiness 패스 상태 기준의 운영용 준비 단계
- 모델 특성/출력 주관성 변경 시 재검증 필수
