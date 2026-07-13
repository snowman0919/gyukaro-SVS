# Training report

Actual SFT completed with OpenMOSS MOSS-TTS-Nano on 64 ASR/script-confirmed real GYU singing phrases. The final checkpoint is `checkpoints/gyu_moss_nano_sft/checkpoint-last` (saved 2026-07-14 03:06:34).

| Setting | Value |
|---|---|
| Epochs / optimizer steps | 3 / 48 |
| Precision / attention | BF16 / SDPA |
| Effective batch / max length | 4 / 512 |
| Learning rate / seed | 1e-5 / 42 |
| Loss step 5 / step 30 | 6.0353 / 5.5230 |

The initial FlashAttention-2 invocation failed because `flash_attn` was unavailable; SDPA rerun completed. This is a Korean-only, small-data vocalizer adaptation. Teacher and SoulX pseudo audio were not used to train its singing decoder.

Machine-readable evidence: `artifacts/reports/training_sft.json` and checkpoint `finetune_config.json`.
