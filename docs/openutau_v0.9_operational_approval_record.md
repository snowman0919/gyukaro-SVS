# OpenUtau v0.9 실사용 운영 승인 기록 (Runtime-path 기준)

생성시각: 2026-07-20T23:05:48.648641+00:00
요청 패키지: `/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau`
실행 패키지: `/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau`
readiness 요약: `/home/kotori9/code/gyukaro/artifacts/reports/openutau_v09/readiness_summary.json`
behavior JSON: `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`
smoke WAV: `/tmp/gyu-v09-operational-check/openutau_v09_smoke.wav`

## 1) 통과 요약
- READY: `true`
- dataset validation: `true`
- pytest: `true`
- verify_v09_runtime_paths: `true`
- operational check: `true`
- 패키지 해시 일치: `true` (`78628cc1b3b51c978c814acc3e15afefcebe08873b49aa3e19d3bd8e1b8a2dc9`)
- smoke 파일 SHA-256: `8cc08abadd0f5fc381ae9caea15fcae04072954ff489a8ca254230a1100c04eb`
- smoke size: `305324` bytes

## 2) 형식 검증
- en: sample_rate=48000, channels=1, seconds=2.12
- ja: sample_rate=48000, channels=1, seconds=2.12
- ko: sample_rate=48000, channels=1, seconds=2.12
- lyric_edit: sample_rate=48000, channels=1, seconds=2.12
- note_pitch_edit: sample_rate=48000, channels=1, seconds=2.12
- style_energetic: sample_rate=48000, channels=1, seconds=2.12
- user_pitch_edit: sample_rate=48000, channels=1, seconds=2.12

## 3) behavioral gates
- energetic_style_changes_audio: true
- ko_en_ja_expected_format: true
- lyric_edit_changes_content: true
- note_edit_changes_pitch: true
- user_pitch_curve_changes_f0: true

## 4) 경로 고정 실행 커맨드
```sh
export GYU_SOULX_RUNTIME_DIR="/home/kotori9/code/gyukaro/.venv-soulx"
export GYU_SOULX_PYTHON="/home/kotori9/code/gyukaro/.venv-soulx/bin/python"
export GYU_SINGER_CACHE="/home/kotori9/code/gyukaro/data/cache"
export GYU_V09_PACKAGE_DIR="/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau"
cd "/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau"
./scripts/openutau_v09_production_readiness.sh
```

## 5) 판단
지정된 runtime 경로 기준으로 오퍼레이션 체크 통과.

## 6) 최신 패키지 내역
- 패키지 보고서: `/home/kotori9/code/gyukaro/artifacts/reports/openutau_v09/behavior.json`
