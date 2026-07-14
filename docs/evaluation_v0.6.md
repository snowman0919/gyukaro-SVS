# v0.6 evaluation

Authoritative prosody evaluation is `artifacts/reports/independent_prosody_evaluation.json` on 24 target-F0-independent score rows. Aggregate F0 correlation / pitch MAE cents: nominal 0.7337 / 137.87; v0.4 0.7328 / 136.03; v0.5 0.7273 / 134.55; experimental v0.6 0.7278 / 135.18. v0.6 improves onset residual (182.61c) and transition contour error (204.24c) versus v0.5, but loses pitch MAE and correlation. It is not claimed as a consistent personalized-prosody win.

KO, EN, and JA v0.6 phrase renders are causal same-score ablations under `artifacts/reports/v06_ablation_identity_*`. On six fixed phrases, student identity minus no identity changes WavLM similarity by mean +0.00196 (bootstrap 95% −0.00158..+0.00561) and ECAPA by +0.00016 (−0.00253..+0.00301). This proves conditioning reaches final audio, not a reliable quality gain.

For two KO phrases, latent-vs-spectral-only dark-style WavLM L2 is 0.002592 and 0.002966. The latent path changes final audio; semantic calibration of every style preset remains weak.
