# v0.6 evaluation

Authoritative prosody evaluation is `artifacts/reports/independent_prosody_evaluation.json` on 24 independently transcribed phrases. Aggregate F0 correlation / pitch MAE cents: nominal verified score 0.7337 / 137.87; v0.4 controller 0.7328 / 136.03; v0.5 real-GYU controller 0.7273 / 134.55; v0.6 verified+high-confidence controller 0.7283 / 134.41. The v0.6 MAE improvement is small and correlation does not beat nominal, so it is not a decisive personalized prosody win.

KO, EN, and JA v0.6 phrase renders were produced at `artifacts/reports/v06_ablation_identity_*`. Four-way same-score identity ablations (none/Fish/Fish+MOSS/student) now use two phrases per language. Student-vs-none changes output embeddings, but it does not establish a reliable identity improvement: WavLM mean +0.00517 (bootstrap 95% −0.00057..+0.01385), ECAPA mean −0.00223. Style compares v0.5 spectral-only, v0.6 spectral-only, and v0.6 latent injection on two KO phrases; latent injection changes WavLM output but semantic calibration remains weak. These are sample-wise evidence, not acceptance evidence.

Package full-path smoke: unpacked `gyu-singer-v0.6-incomplete.zip` ran `run.sh` with the pinned local cache and produced a 48 kHz mono WAV. This verifies package/runtime wiring; it is not a claim that upstream model weights are self-contained.
