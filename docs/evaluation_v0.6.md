# v0.6 evaluation

Authoritative prosody evaluation is `artifacts/reports/independent_prosody_evaluation.json` on 24 independently transcribed phrases. Aggregate F0 correlation / pitch MAE cents: nominal verified score 0.6846 / 56.75; v0.4 controller 0.6829 / 58.49; v0.5 real-GYU controller 0.6839 / 59.48; v0.6 verified+high-confidence controller 0.6844 / 58.27. The v0.6 MAE improvement is small and correlation does not beat nominal, so it is not a decisive personalized prosody win.

KO, EN, and JA v0.6 phrase renders were produced at `artifacts/reports/v06_{ko,en,ja}.wav`. Same-score identity ablation changes WavLM cosine to the GYU reference from 0.67271 (enabled) to 0.64184 (zero), and latent style zeroing changes it to 0.60861. These prove conditioning influence, not yet a perceptual quality win. ECAPA was not run in this environment.
