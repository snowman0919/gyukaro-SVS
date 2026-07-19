# OpenUtau v0.9 실사용 런북

## 목적
지정된 runtime 경로에서 OpenUtau v0.9를 바로 실행/검증하여 운영 반출 가능한지 확인합니다.

## 사전 조건
- `/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau` 존재
- `.venv-soulx` 존재 (`/home/kotori9/code/gyukaro/.venv-soulx`)
- `data/cache` 존재 (`/home/kotori9/code/gyukaro/data/cache`)
- OpenUtau 소스/빌드 경로: `/tmp/OpenUtau`

## 1) 환경 고정

```sh
export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx
export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python
export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache
export GYU_V09_PACKAGE_DIR=/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau
export GYU_V09_READINESS_OUTPUT_DIR=/tmp/gyu-v09-runtime-readiness
export GYU_SMOKE_OUTPUT_DIR=/tmp/gyu-v09-runtime-smoke
export GYU_SMOKE_PORT=8780
export OPENUTAU_REPO=/tmp/OpenUtau
```

## 2) 원클릭 READY 체크 (권장)

```sh
cd /home/kotori9/code/gyukaro
/home/kotori9/code/gyukaro/scripts/openutau_v09_ready_check.sh
```

성공 조건
- 출력에 `READY= True` 표시
- `/tmp/gyu-v09-runtime-readiness/readiness_summary.json` 에서:
  - `package.hash_match == true`
  - `verify_v09_runtime_paths.smoke_status_0 == true`
  - `operational_check.pass == true`
  - `dataset_validation.pass == true`
  - `pytest_check.pass == true`
  - `operational_check.gates` 모든 항목 `true`

## 3) 운영 smoke 3단계

```sh
/home/kotori9/code/gyukaro/scripts/openutau_v09_operational_check.sh /home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau
```

성공 조건
- `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json` 파일 생성
- `pass: true`
- 각 출력 WAV가 48kHz mono, 비어있지 않음

## 4) 패키지 smoke (OpenUtau 통합)

```sh
cd /home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau
export GYU_SOULX_RUNTIME_DIR=${GYU_SOULX_RUNTIME_DIR:-/home/kotori9/code/gyukaro/.venv-soulx}
export GYU_SOULX_PYTHON=${GYU_SOULX_PYTHON:-/home/kotori9/code/gyukaro/.venv-soulx/bin/python}
export GYU_SINGER_CACHE=${GYU_SINGER_CACHE:-/home/kotori9/code/gyukaro/data/cache}
export OPENUTAU_REPO=${OPENUTAU_REPO:-/tmp/OpenUtau}
export GYU_SMOKE_OUTPUT_DIR=/tmp/gyu-v09-package-local
scripts/openutau_v09_runtime_smoke.sh
```

성공 조건
- 마지막줄 `openutau v0.9 runtime smoke done`
- `/tmp/gyu-v09-package-local/openutau_v09_smoke.wav` 생성
- OpenUtau resident integration test가 `Passed: 1` 출력

## 5) 운영 실패 시 점검 순서
1. 경로가 절대경로인지 확인 (`GYU_V09_PACKAGE_DIR`, `GYU_SOULX_RUNTIME_DIR`, `GYU_SOULX_PYTHON`, `GYU_SINGER_CACHE`)
2. `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json` 의 `gates` 실패 항목 확인
3. `/tmp/gyu-v09-package-local/openutau_v09_smoke.log`/`serve.log`에서 에러 라인 확인
4. `/tmp/gyu-v09-package-local/openutau_resident_test.log`에서 OpenUtau 테스트 실패 원인 확인
5. 다시 1번부터 재실행

## 6) 운영 승인 체크리스트(요약)

| 항목 | 기대값 | 실제값 |
|---|---|---|
| READY 체크 | READY= True |  |
| 패키지 해시 | hash_match true |  |
| runtime 경로 검증 | smoke_status_0 = 0, 파일 크기 > 0 |  |
| 운영 동작 게이트 | pass = true |  |
| 게이트 상세 | 5개 항목 모두 true |  |
| 운영 smoke | `openutau v0.9 runtime smoke done` |  |
| 통합 테스트 | Passed: 1, Failed: 0 |  |
| 오디오 출력 | ko/en/ja 48kHz mono, WAV 존재 |  |

비고: `/tmp/gyu-v09-runtime-readiness/readiness_summary.json`, `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`를 보존해 감사 근거로 첨부하세요.

## 7) 운영 승인 기록 템플릿

```
점검 일시: ____-__-__ __:__
점검자: ____________________
호스트: ____________________
브랜치/빌드: ____________________

1) READY 체크
- readiness_summary 경로: ___________________________________
- READY=True: _____ (Y/N)
- 실패 항목(있다면): _______________________________________

2) smoke/동작 확인
- pass: _____ (Y/N)
- ko/en/ja WAV 존재: _____ (Y/N)
- ko/en/ja 샘플레이트: _____ Hz, 채널: _____

3) OpenUtau 통합 테스트
- openutau_resident_test.log 보존 경로: ____________________
- Passed/Failed: ____________________

4) 비고
- 특이 이슈/메모: ___________________________________________
```

## 8) 실사용 단일 실행(권장)

아래 한 줄로 지정 경로 기준 전체 체크를 실행합니다.

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_full_runtime_readiness.sh
```

기록 산출물:
- `openutau_v09-readiness-summary`(기본: `artifacts/reports/openutau_v09/readiness_summary.json`)
- `openutau_v09-operational-behavior`(기본: `$GYU_SMOKE_OUTPUT_DIR/openutau_v09_operational_behavior.json`, 기본값 `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`)
- `openutau_v09-smoke.wav`(기본: `$GYU_SMOKE_OUTPUT_DIR/openutau_v09_smoke.wav`, 기본값 `/tmp/gyu-v09-operational-check/openutau_v09_smoke.wav`)
