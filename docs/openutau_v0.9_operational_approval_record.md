# OpenUtau v0.9 실사용 운영 승인 기록 (Runtime-path 기준)

생성시각: 2026-07-19T21:32:10Z
패키지: `/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau`
readiness 요약: `/home/kotori9/code/gyukaro/artifacts/reports/openutau_v09/readiness_summary.json`
behavior JSON: `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`
smoke WAV: `/tmp/gyu-v09-operational-check/openutau_v09_smoke.wav`

## 1) 통과 요약
- `READY`: `true`
- dataset validation: `pass`
- pytest: `pass` (`5 passed, 3 warnings`)
- verify_v09_runtime_paths: `pass`
- operational check: `pass`
- behavioral gates: all true (`ko/en/ja` format, note edit, user pitch curve, lyric edit, energetic style)
- 패키지 해시 일치: `ff8890c703af76439da135a8c4738706faa4e0bd330cf4f8bbc1241739e99e10`
- smoke 파일 SHA-256: `8cc08abadd0f5fc381ae9caea15fcae04072954ff489a8ca254230a1100c04eb`

## 2) 형식 검증
- KO/EN/JA/편집 항목 출력 샘플레이트: 48k
- 채널: mono
- 길이: 약 2.12초

## 3) 경로 고정 실행 커맨드
```sh
export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx
export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python
export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_full_runtime_readiness.sh
```

## 4) 판단
지정된 runtime 경로 기준으로 **오퍼레이션 체크가 재현 가능한 범위에서 통과**했습니다.

OpenUtau 통합 smoke(`openutau_v09` 패키지)도 동작하고, `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`에 제어별 gate pass 결과가 기록됩니다.
