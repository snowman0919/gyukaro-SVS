# GYU Singer v1 experimental

Overall status: packaged neural v1-experimental
Packaged v1: `gyu-singer-v1-experimental.zip`
Package path: `artifacts/package/gyu-singer-v1-experimental.zip`
Package SHA-256: `e345663d08ddb7f13feff0684df81893f4972552d5c93975a779702ef33a81a6`
Best checkpoint: `checkpoints/gyu_moss_nano_sft/checkpoint-last`
Renderer status: CLI plus resident localhost HTTP daemon; fine-tuned neural backend default
Editor integration status: OpenUtau score-export bridge protocol
Korean status: real GYU SFT and score render validated
English status: foundation-model cross-lingual render validated; experimental after KO-only SFT
Japanese status: foundation-model cross-lingual render validated; experimental after KO-only SFT

## What actually works

132 real recordings are indexed; 76 phrase alignments are ASR/script confirmed; 64 real D/E singing phrases were tokenized and used for SFT. `gyu-singer render` accepts score JSON and renders 48 kHz mono WAV with lyric conditioning, MIDI pitch, timing, and dynamics. Clean-package smoke rendered a 6.45 s WAV without external model cache.

## Measured results

SFT: 64 examples, 3 epochs, 48 optimizer steps, BF16, global batch 4, max length 512. Logged loss: 6.0353 at step 5, 5.5230 at step 30, final checkpoint at step 48. KO/EN/JA finetuned renderer samples are each 48 kHz mono, 6.45 s.

## Architecture chosen

Fine-tuned MOSS-TTS-Nano acoustic-token model plus GYU reference conditioning, then note-by-note pitch/time DSP. See `architecture.md`.

## Teacher models actually used

MOSS-TTS-Nano KO/EN/JA clone pilot was run and acoustically filtered. Fish S2 Pro, Higgs TTS 3 4B, and MOSS Local Transformer v1.5 each produced a separate Korean GYU-reference clone pilot; none of their generated audio was admitted to training. See `teacher_report.md`.

## SVS systems actually inspected

TCSinger 2, FM-Singer, TechSinger, SoulX-Singer, YingMusic-Singer Plus, and OpenVPI DiffSinger; see `svs_review.md`.

## Important compromises

SFT corpus is Korean-only and tiny. Output is score-controlled vocalization, not full phoneme-to-note neural SVS. Empty isolated-prompt Nano generations fall back to real GYU reference audio; this is recorded in runtime code and must be eliminated by fuller acoustic training.

## Run v1

```sh
cd artifacts/package/gyu-singer-v1-experimental
sh install.sh
sh run.sh
```

## Next highest-value improvements

Run full 100×3 teacher benchmark with speaker/ASR gates, add EN/JA supervised or high-confidence pseudo-singing data, then replace DSP note conversion with a pretrained flow-matching SVS acoustic decoder.
