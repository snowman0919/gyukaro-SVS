# OpenUtau v0.9 integration

## Architecture

The supported integration is a small maintained fork overlay against current official OpenUtau commit `27573ac5c888d927119d5f65a207312d79194b1f`. Current OpenUtau exposes `IRenderer` and renderer registration inside `OpenUtau.Core`, but no external renderer-registration API. A pinned three-file registration patch plus the self-contained `GyuSingerRenderer.cs` is therefore smaller and more stable than pretending a standalone exporter is native integration.

`GYU-SINGER` is a built-in virtual singer. Its dummy OTO entries only allow normal `RenderPhrase` construction; they emit no audio. OpenUtau sends each multi-note phrase to a single resident `gyu-singer-v0.8` process, receives one 48 kHz WAV, performs the single editor-standard 44.1 kHz conversion, then caches and plays/exports the phrase.

```text
OpenUtau RenderPhrase
  -> notes/lyrics/phones/tempo/final pitch and expression curves
  -> GYU protocol v2 JSON
  -> resident v0.8 (v0.5 prosody + v0.7 identity/style + SoulX)
  -> 48 kHz phrase WAV
  -> OpenUtau cache/playback/export
```

The static HTTP client and resident workers avoid reloading OmniVoice or SoulX per phrase. SHA-256 of the complete request is the cache key, so note, lyric, pitch, expression, and style edits invalidate the cached WAV. Network/model errors are surfaced as a renderer failure naming `GYU_RENDERER_URL` and `/health`.

## Mapped controls

- Notes, tuning, lyrics, tempo, and generated phonemes are timed from the actual `RenderPhrase`.
- OpenUtau `phrase.pitches` already combines note pitch, portamento, note vibrato, and PITD. It is converted to a score-relative semitone residual and remains authoritative after learned GYU prosody.
- DYN, BREC, and TENC map to dynamics, breathiness, and tension.
- GYUS values 0..5 map to neutral, soft, breathy, energetic, relative C, and relative B.
- Unsupported as separate controls: brightness curve, an independent vibrato-depth curve, and expressions not listed above.

Only breathy and energetic have repeated acoustic-proxy direction evidence. Soft, dark, and bright are relative latent controls; no calibrated semantic claim is made.

## Evidence

The overlay compiles on .NET 8 against the pinned official tree. The native package checks run through 4 tests (package checks + OpenUtau `OpenUtau.Test` mapping integration), and a separate resident integration test executes the real C# `IRenderer.Render` call and receives non-silent 44.1 kHz phrase samples from the running v0.8 backend.

The reproducible behavioral report is `artifacts/reports/openutau_v09/behavior.json`:

- KO/EN/JA: 48 kHz mono, 2.12 seconds each.
- +2-semitone note edit: RMVPE median shift +200.41 cents.
- +1-semitone PITD edit: RMVPE median shift +92.52 cents.
- lyric edit: Whisper changes from “하늘빛 노래” to “사랑을 담아서 내준 내 마음”; waveform RMS difference 0.118047.
- neutral vs energetic: final RMS 0.078379 vs 0.079411 with score, lyrics, identity, and pitch fixed.

This proves causal editor behavior, not a listening-quality preference. The energetic effect is small but nonzero and agrees with the prior held-out energy proxy direction.

## Character metadata bundle (OpenUtau singer discovery)

`scripts/package_v09.py` now includes `openutau_character_library/GYU-SINGER/` in the archive.
It contains:

- `character.txt`
- `character.yaml`
- `README.md`
- portrait image files copied from `gyu/*.png`
- `as/urls.txt` when present

OpenUtau uses this bundle for singer metadata/portrait discovery and configuration.
It does **not** replace the maintained OpenUtau renderer path:
actual phrase rendering is still performed by the forked GYU renderer path described above.

## Install and test

See `integrations/openutau/README.md`. The essential commands are:

```sh
git clone https://github.com/stakira/OpenUtau.git
git -C OpenUtau checkout 27573ac5c888d927119d5f65a207312d79194b1f
./integrations/openutau/install_fork.sh OpenUtau
dotnet build OpenUtau/OpenUtau.csproj -c Release
# If you run from repository source checkout root, use <repo-root>/data/cache
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python  # optional when auto-discovery does not apply
# or
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-root  # optional: directory containing .venv/bin/python or bin/python
export GYU_RENDERER_URL=http://127.0.0.1:8765/render
./serve.sh 8765
```

For source-tree developers (before packaging), start the renderer with:

```sh
./serve.sh 8765
```

Note: if `dotnet` is not on PATH, these helpers can still run with local runtimes
such as `/tmp/dotnet/dotnet`.

Then open `examples/openutau_v09.ustx`. All three tracks use the phrase renderer. The headless `bridge.py` remains only a debugging/export path and is not counted as the v0.9 integration.

## Runtime smoke

For repeatable smoke checks (recommended), run:

```sh
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
# If you run from repository source checkout root, use <repo-root>/data/cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python
# or
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-root
export OPENUTAU_REPO=/absolute/path/to/patched/OpenUtau
export GYU_SMOKE_PORT=8765
export GYU_SMOKE_OUTPUT_DIR=/tmp/gyu-v09-smoke
./scripts/openutau_v09_runtime_smoke.sh
```

The script runs: resident boot, `/health`, `/model`, bridge render, and resident integration test.


실사용 one-click 보조진입은 아래 한 줄로 시작됩니다.

```sh
./scripts/openutau_v09_runtime.sh start
./scripts/openutau_v09_runtime.sh status
./scripts/openutau_v09_runtime.sh stop
```

(필요 시 경로는 환경변수로 덮어쓰기 가능: `GYU_SOULX_RUNTIME_DIR`, `GYU_SOULX_PYTHON`, `GYU_SINGER_CACHE`, `GYU_V09_PACKAGE_DIR`, `GYU_SMOKE_PORT`)

## 실사용 실행 체크리스트 (빠른 점검)

1) 서비스 부팅
```sh
cd /path/to/gyu-singer-v0.9-openutau
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-soulx
./serve.sh 8765
```

2) 기본 헬스 + 메타데이터 확인
```sh
curl -fsS http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/model
```

3) KO/EN/JA 3개 트랙 브릿지 렌더 체크 (짧은 버전)
```sh
python integrations/openutau/bridge.py examples/openutau_v09.ustx --language ko \
  --output /tmp/openutau-v09-request.json --render-url http://127.0.0.1:8765 --wav /tmp/openutau-v09-smoke.wav
```

4) OpenUtau resident 통합 테스트
```sh
cd /tmp/OpenUtau
GYU_RENDERER_URL=http://127.0.0.1:8765/render dotnet test OpenUtau.Test/OpenUtau.Test.csproj -c Release --filter FullyQualifiedName~GyuSingerResidentIntegrationTest
```

5) 파이프라인 게이트
```sh
python -m pytest tests/test_openutau_diffsinger_package.py tests/test_openutau_native_evaluation.py
```

모든 단계가 통과되면 운영 환경에서 곧바로 사용 가능합니다.

### 실사용 최종 점검(권장 단일 명령군)

아래는 현재 검증 기준으로 통과한 최소 운영 체크입니다.

```sh
cd /home/kotori9/code/gyukaro
# 지정 런타임 패키지와 캐시 경로가 고정된 실행 체크(권장)
export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx
export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python
export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache
./scripts/openutau_v09_full_runtime_readiness.sh
./scripts/openutau_v09_ops.sh readiness
```

또는 운영 반출에서 권장되는 최종 명령:

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_production_readiness.sh
```

시스템d 자동기동이 필요한 경우:
```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_ops.sh start-systemd
./scripts/openutau_v09_ops.sh systemd-restart
./scripts/openutau_v09_ops.sh systemd-quickstart
./scripts/openutau_v09_ops.sh systemd-status
```

또는 승인 문서까지 동시에 갱신하려면:

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_collect_approval_record.sh
```

또는, 수동 체크(동일 동작):

```sh
cd /home/kotori9/code/gyukaro
export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx
export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python
export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache
export OPENUTAU_REPO=/tmp/OpenUtau
export GYU_SMOKE_OUTPUT_DIR=/tmp/gyu-v09-smoke

# 1) 패키지 경로 기준 고정 런타임 경로 검사
./scripts/verify_v09_runtime_paths.sh /home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau

# 기대 출력:
# smoke_status=0
# render_size=1428524

# 2) 패키지 스크립트의 phrase-level 동작 검증
cd /home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau
setsid env GYU_SOULX_RUNTIME_DIR=$GYU_SOULX_RUNTIME_DIR \
  GYU_SOULX_PYTHON=$GYU_SOULX_PYTHON \
  GYU_SINGER_CACHE=$GYU_SINGER_CACHE \
  ./serve.sh 8780 >/tmp/v09-ready-serve.log 2>&1 < /dev/null &
sleep 2
cd /home/kotori9/code/gyukaro
PYTHONPATH=src python artifacts/package/gyu-singer-v0.9-openutau/scripts/test_openutau_v09_behavior.py \
  --render-url http://127.0.0.1:8780 \
  --output /tmp/gyu-v09-ready-behavior.json
cat /tmp/gyu-v09-ready-behavior.json
```

`openutau_v09_operational_check.sh`는 상태 로그를 stderr로 출력하고 JSON만 stdout으로 출력하므로, JSON 파싱은 `$GYU_SMOKE_OUTPUT_DIR/openutau_v09_operational_behavior.json` (기본값 `/tmp/gyu-v09-operational-check/openutau_v09_operational_behavior.json`)로 하세요.

```sh
GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx \
GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python \
GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache \
./scripts/openutau_v09_operational_check.sh /home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau >/tmp/openutau_v09_operational_check_stdout.log
jq -r '(.pass and (.gates | all(.))) | tostring' "${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}/openutau_v09_operational_behavior.json"
```

`pass: true`가 출력되고 `${GYU_SMOKE_OUTPUT_DIR:-/tmp/gyu-v09-operational-check}/openutau_v09_operational_behavior.json`에서 `ko/en/ja`가 48k mono인지를 확인하면 OpenUtau 경로는 실사용 조건을 만족한 상태입니다.

For a single fixed-runtime-path check (source-tree and packaged tree), use:

```sh
./scripts/verify_v09_runtime_paths.sh /path/to/gyu-singer-v0.9-openutau
```

or one-command operational snapshot:

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_ready_check.sh
```

필요 시 패키지 테스트(5개)까지 포함한 오퍼레이션 단일 체크는 아래입니다.

```sh
cd /home/kotori9/code/gyukaro
export GYU_OPS_RUN_PKG_TESTS=1
./scripts/openutau_v09_ops_check.sh
```

승인 기록은 다음 경로로 갱신됩니다.

```sh
/home/kotori9/code/gyukaro/artifacts/reports/openutau_v09/operational_approval_record.md
```

환경이 다른 경로를 쓸 때는 아래 변수로 덮어쓸 수 있습니다.

```sh
export GYU_V09_PACKAGE_DIR=/path/to/gyu-singer-v0.9-openutau
export GYU_V09_READINESS_OUTPUT_DIR=/path/to/artifacts/reports/openutau_v0.9
./scripts/openutau_v09_ready_check.sh
```

운영 승인 리포트에는 두 경로가 모두 기록됩니다.

- 요청 패키지: 실행 전 `GYU_V09_PACKAGE_DIR` 값(또는 기본 경로)
- 실행 패키지: 실제 readiness에서 사용한 경로(`.zip`이면 준비/해제 후 경로)

Equivalent manual flow:

```sh
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python
# or
export GYU_SOULX_RUNTIME_DIR=/absolute/path/to/.venv-root
./serve.sh 8765 >/tmp/gyu-singer-v0.9-serve.log 2>&1 &
sleep 2
curl -s http://127.0.0.1:8765/health
python integrations/openutau/bridge.py examples/openutau_v09.ustx --language ko \
  --output /tmp/openutau-v09-request.json --render-url http://127.0.0.1:8765 --wav /tmp/openutau-v09-smoke.wav
  # --render-url may be the base service URL or .../render endpoint

cd /tmp/OpenUtau
GYU_RENDERER_URL=http://127.0.0.1:8765/render dotnet test OpenUtau.Test/OpenUtau.Test.csproj -c Release --filter FullyQualifiedName~GyuSingerResidentIntegrationTest

cd /path/to/gyukaro
python -m pytest tests/test_openutau_diffsinger_package.py tests/test_openutau_native_evaluation.py
```


운영팀용 상세 런북: [docs/openutau_v0.9_runbook.md](docs/openutau_v0.9_runbook.md)

## 실사용 Go-Live 요약 (권장)

현재 고정 경로 기준으로 아래 명령을 실행했을 때 모두 통과되어야 실사용 투입 조건을 만족합니다.

```sh
cd /home/kotori9/code/gyukaro
export GYU_SOULX_RUNTIME_DIR=/home/kotori9/code/gyukaro/.venv-soulx
export GYU_SOULX_PYTHON=/home/kotori9/code/gyukaro/.venv-soulx/bin/python
export GYU_SINGER_CACHE=/home/kotori9/code/gyukaro/data/cache
export GYU_V09_PACKAGE_DIR=/home/kotori9/code/gyukaro/artifacts/package/gyu-singer-v0.9-openutau
./scripts/openutau_v09_production_readiness.sh
```

필수 확인값(문서 갱신 기준):

- `READY: true`
- dataset/pytest/runtime/operational 모두 pass
- `openutau_v0.9` 패키지 hash match
- smoke WAV: `/tmp/gyu-v09-operational-check/openutau_v09_smoke.wav`
- smoke SHA: `8cc08abadd0f5fc381ae9caea15fcae04072954ff489a8ca254230a1100c04eb`
- bridge 렌더(ko/en/ja) WAV 생성 성공 (`pcm_s24le`, `48kHz`, `1ch`, `2.120000s`)

최종 증빙은 고정 산출물에서 관리합니다.

- `/tmp/openutau_v09_go_live_evidence.json`
- `/tmp/openutau_v09_operational_check_stdout.log`
- `/home/kotori9/code/gyukaro/docs/openutau_v0.9_go_live_evidence.md`
- `/home/kotori9/code/gyukaro/docs/openutau_v0.9_operational_approval_record.md`

실사용 승인 문서까지 통합 생성하려면 아래 한 줄을 권장합니다.

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_oneclick_operational_check.sh
```

`openutau_v09_oneclick_operational_check.sh`는 `openutau_v09_collect_approval_record.sh`를 통해
실행 로그와 함께 승인 문서를 갱신하며, `요청 패키지`/`실행 패키지` 경로가 같지 않아도 추적됩니다.

## 실사용 원클릭 런치(선택)

고정 경로 기준으로 바로 운영 서비스를 띄우고 기본 헬스를 검사하려면:

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_go_live.sh
```

서비스 중지/상태 확인:

```sh
./scripts/openutau_v09_go_live.sh 8765 --status   # 실행 중 확인
./scripts/openutau_v09_go_live.sh 8765 --stop     # 종료
```

기록:
- PID: `/tmp/gyu-v09-go-live.pid`
- 로그: `/tmp/gyu-v09-go-live.log`
- 헬스 결과: `/tmp/gyu-v09-go-live-health.json`
- 시작 체크: `/tmp/gyu-v09-go-live-check.json`

## systemd 자동실행(선택)

부팅 자동기동/감시가 필요하면 다음으로 유저 서비스 파일을 생성할 수 있습니다.

```sh
cd /home/kotori9/code/gyukaro
./scripts/openutau_v09_systemd_unit.sh 8765 gyu-openutau-v0.9.service print
# 출력된 내용을 user unit으로 저장하거나 직접 설치
./scripts/openutau_v09_systemd_unit.sh 8765 gyu-openutau-v0.9.service apply

# 기동/중지
systemctl --user start gyu-openutau-v0.9.service
systemctl --user stop gyu-openutau-v0.9.service

# 상태 확인
systemctl --user status gyu-openutau-v0.9.service
```

설치 시 생성/기록 위치:

- unit: `~/.config/systemd/user/gyu-openutau-v0.9.service`
- 서비스 로그: `artifacts/reports/openutau_v09/openutau-v09-service.log`
