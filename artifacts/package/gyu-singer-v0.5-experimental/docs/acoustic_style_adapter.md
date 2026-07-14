# GYU acoustic-style adapter (v0.5)

`GyuAcousticStyleAdapter` predicts a 257-bin spectral log-gain from GYU reference
features, preset, and continuous controls. The v0.5 renderer applies this gain
to OmniVoice phrase content before the SoulX-Singer decode; it is not an F0-only
branch. The teacher-distilled student identity projection perturbs the reference
representation used by this adapter at inference.

Checkpoint: `checkpoints/gyu_acoustic_style_adapter_v0.5.pt`.
Training evidence: `artifacts/reports/acoustic_style_adapter_training.json`.
Known limitation: spectral adapter is compact and not perceptually calibrated.
