# v0.6 evaluation

Authoritative prosody evaluation is `artifacts/reports/independent_prosody_evaluation.json` on 24 independently transcribed phrases. Aggregate F0 correlation / pitch MAE cents: nominal verified score 0.6846 / 56.75; v0.4 controller 0.6829 / 58.49; v0.5 real-GYU controller 0.6839 / 59.48; v0.6 verified+high-confidence controller 0.6844 / 58.27. The v0.6 MAE improvement is small and correlation does not beat nominal, so it is not a decisive personalized prosody win.

KO, EN, and JA v0.6 phrase renders were produced at `artifacts/reports/v06_{ko,en,ja}.wav`. Four-way same-score identity ablations (none/Fish/Fish+MOSS/student) change both WavLM and ECAPA distributions; across three languages student-vs-none means are +0.01080 WavLM and +0.00067 ECAPA. Latent style zeroing also changes the KO output. These are sample-wise evidence, not significance claims.

Package full-path smoke: unpacked `gyu-singer-v0.6-incomplete.zip` ran `run.sh` with the pinned local cache and produced a 48 kHz mono WAV. This verifies package/runtime wiring; it is not a claim that upstream model weights are self-contained.
