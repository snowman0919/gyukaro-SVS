# GYU Singer v1 experimental

Overall status: packaged experimental hybrid runtime
Packaged v1: `gyu-singer-v1-experimental.zip`
Package path: `artifacts/package/gyu-singer-v1-experimental.zip`
Package SHA-256: `21582fb38c0131e594f98ec7213579cd5808f207c337e9e56ac56bfbb6b684dc`
Best checkpoint: `checkpoints/gyu_hybrid_v0.2.pt`
Renderer status: phrase-level CLI and resident HTTP (`GET /health`, `GET /model`, `POST /render`)
Editor integration status: executable USTX score exporter/HTTP bridge
Korean status: real-anchor trained; experimental quality
English status: frontend/render exercised; experimental
Japanese status: frontend/render exercised; experimental

## What actually works

132 source recordings were indexed and 76 real GYU supervision rows prepared. Hybrid model renders entire phrases in one conditional-flow generation then decodes them with a frozen pretrained codec. Clean package smoke generated 48 kHz mono WAV using only packaged runtime and model files. Baseline source-loop and per-note TTS/DSP paths are not hybrid SVS.

## What does not work

Real-anchor score timing is inferred, not verified singing score annotation. No pseudo-singing row passed a multilingual admission gate; one English pilot remains evaluation-only. No listening study, intelligibility score or pitch-accuracy metric proves production singing quality. OpenUtau bridge is not a native renderer plugin.

## Measured results

160 steps on GB10: final total loss 2.353837, flow 2.289662, pitch 0.057003, teacher representation 0.408833. 60/5/5 real train/valid/test rows; 633 teacher representation rows. Package smoke: 48 kHz mono, 307200 frames.

## Architecture chosen

TriSinger phrase-level conditional flow combines language-aware phonemes, score, blurred boundaries, timbre, style and explicit pitch conditions. MOSS audio tokenizer decoder is frozen. See `docs/architecture_v2.md`.

## Teacher models actually used

Fish S2 Pro, Higgs TTS 3 4B and MOSS Local Transformer v1.5 generated/evaluated teacher rows; retained rows supervise representation loss only.

## SVS systems actually inspected

TCSinger 2, FM-Singer, TechSinger, SoulX-Singer, YingMusic-Singer Plus and OpenVPI DiffSinger.

## Training completed

`PYTHONPATH=src python scripts/prepare_hybrid_data.py`; `scripts/cache_hybrid_latents.py`; `scripts/train_hybrid.py --steps 160`.

## Important compromises

Small real corpus, inferred bootstrap scores, compact model and no accepted multilingual pseudo-singing data. Quality claims are deliberately limited to executable rendering.

## Exact commands to run v1

```sh
cd artifacts/package/gyu-singer-v1-experimental
sh install.sh
sh run.sh
```

## Next highest-value improvements

Create verified musical score alignments; admit a quality-gated multilingual pseudo-singing set; add validation metrics/listening evaluation; implement native OpenUtau renderer integration.
