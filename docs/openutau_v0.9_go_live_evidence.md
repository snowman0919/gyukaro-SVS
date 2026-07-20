# OpenUtau v0.9 Release v0.9v Go-Live Evidence

생성 시각(UTC): `2026-07-20T22:52:20+00:00`

## 1) 릴리스 대상
- 패키지: `artifacts/package/gyu-singer-v0.9v-openutau.zip`
- 패키지 SHA-256: `78628cc1b3b51c978c814acc3e15afefcebe08873b49aa3e19d3bd8e1b8a2dc9`
- 동작 기준 패키지: `artifacts/package/gyu-singer-v0.9-openutau`

## 2) 운영/패키지 게이트
- READY: `true`
- dataset validation: `PASS (132)`
- pytest(패키지): `PASS (5 passed)`
- runtime path check: `PASS`
- operational check: `PASS`
- package hash match: `PASS`
- smoke WAV SHA-256: `8cc08abadd0f5fc381ae9caea15fcae04072954ff489a8ca254230a1100c04eb`
- smoke 파일 길이: `305324` bytes

## 3) 형식/동작 검증
- ko/en/ja 렌더 결과: `48kHz / mono / 2.12s`
- note_pitch_edit/user_pitch_edit/style_energetic/lyric_edit 출력: `48kHz / mono / 2.12s`
- behavioral gates: `ko_en_ja_expected_format, note_edit_changes_pitch, user_pitch_curve_changes_f0, lyric_edit_changes_content, energetic_style_changes_audio`

## 4) 실행 환경
- `GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx`
- `GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python`
- `GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache`
- `GYU_V09_PACKAGE_DIR=/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau`
- `openutau_v09_operational_behavior` 경로: `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`
- `openutau_v09_operational_check` 로그: `/tmp/gyu-v09-operational-check/openutau_v09_operational_check_stdout.log`

## 5) OpenUtau 연동 체크
- OpenUtau 8.0.422 + .NET 환경에서 resident test(1건) PASS
- package smoke의 `native`/`openutau` 경로 모두 통과

## 6) 비고
- 현재 상태는 **릴리스 게이트 통과 증적**이 있으나, 실제 운영 투입은 사용자 승인 기준(필요 시 사람 청취) 완료 후 진행 권장.
