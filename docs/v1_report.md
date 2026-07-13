# GYU Singer v1 experimental

Overall status: packaged v1-experimental real-anchor renderer
Packaged v1: `gyu-singer-v1-experimental.zip`
Package path: `artifacts/package/gyu-singer-v1-experimental.zip`
Package SHA-256: `99167173da40ff464c2cb3d6a440237ba6b8e66d183d1e62602f7ce5510c4d0c`
Best checkpoint: `checkpoints/gyu_v1_experimental.npz`
Renderer status: CLI and resident localhost HTTP daemon implemented
Editor integration status: score-export bridge protocol only
Korean status: pitch-controlled GYU timbre; lyrics not intelligible
English status: experimental protocol input only; no phoneme model
Japanese status: experimental protocol input only; no phoneme model

## What works

Dataset index/PCM masters, conservative anchor manifest, real-GYU adapted loop checkpoint, explicit MIDI pitch/duration rendering, three rendered score samples, package archive, and clean package smoke test.

## What does not work

No manual transcript/forced alignment; no teacher generation; no pretrained neural SVS adaptation; no lyric-conditioned intelligibility; no native OpenUtau renderer.

## Measured results

Validation asserts 132 continuous source indices and 48 kHz mono PCM. Render smoke output is 48 kHz mono PCM24. Eight loops are selected only from voiced, unclipped real anchors.

## Architecture chosen

Real-anchor source-loop renderer described in `architecture.md`, selected because it produces real speaker audio today without pretending unverified text can train trilingual neural singing.

## Teacher models actually used

None. Three official candidates were evaluated for interface/license feasibility only; see `teacher_report.md`.

## SVS systems actually inspected

TCSinger 2, FM-Singer, TechSinger, SoulX-Singer, YingMusic-Singer Plus, and OpenVPI DiffSinger; see `svs_review.md`.

## Training completed

Deterministic real-anchor selection/adaptation complete. No neural training claimed.

## Important compromises

Text ignored acoustically; this is explicitly experimental. Human transcript alignment and pretrained SVS adaptation are highest-value next work.

## Run v1

```sh
PYTHONPATH=src python -m gyu_singer.cli --model checkpoints/gyu_v1_experimental.npz render examples/korean.json --output out.wav
PYTHONPATH=src python -m gyu_singer.cli --model checkpoints/gyu_v1_experimental.npz serve --port 8765
```

## Next highest-value improvements

Manually review `script_alignment.jsonl`, obtain verified Korean labels/phonemes, benchmark all three teachers, then adapt a license-compatible pretrained SVS acoustic model.
