# OpenUtau v0.9 실사용 경로 Go-Live 증빙

생성 시각(UTC): `2026-07-20T02:11:50.698386+00:00`

## 1) 런타임 게이트
- READY: `true`
- dataset_validation_pass: `true`
- pytest_pass: `true`
- runtime_smoke_pass: `true`
- operational_pass: `true`
- package_hash_match: `true`
- package hash: `0d7932b112493f023edfb14481d44db14a6db6340cad5e0b7b2e9e520ccbaa8a`
- smoke WAV SHA: `8cc08abadd0f5fc381ae9caea15fcae04072954ff489a8ca254230a1100c04eb`
- smoke WAV: `/tmp/gyu-v09-operational-check/openutau_v09_smoke.wav`

## 2) behavioral gates
ko_en_ja_expected_format: `True`, note_edit_changes_pitch: `True`, user_pitch_curve_changes_f0: `True`, lyric_edit_changes_content: `True`, energetic_style_changes_audio: `True`

## 3) bridge 엔드투엔드 (ko/en/ja)
- ko: `/tmp/gyu-v09-bridge-ko-actual.wav`
- en: `/tmp/gyu-v09-bridge-en-actual.wav`
- ja: `/tmp/gyu-v09-bridge-ja-actual.wav`

각각 형식: `pcm_s24le`, `sample_rate=48000`, `channels=1`, `duration=2.120000`

## 4) 고정 경로 실행 환경
- `GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx`
- `GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python`
- `GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache`
- `export GYU_V09_PACKAGE_DIR=/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau`
- `./scripts/openutau_v09_production_readiness.sh`

## 5) 권고
- 운영 반영은 사람 청취 승인과 함께 진행하세요.
- 필요 시 `docs/openutau_v0.9.md`의 'Runbook' 절차로 재확인하세요.
