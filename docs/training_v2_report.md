# Training v0.2 report

Run: `PYTHONPATH=src python scripts/train_hybrid.py --steps 1200` on CUDA/NVIDIA GB10. Optimizer AdamW, LR `2e-4`, batch 1, no accumulation, default torch precision. Checkpoint: `checkpoints/gyu_hybrid_v0.2.pt`; report: `artifacts/reports/hybrid_training.json`.

Data: 60 real anchors plus 2 accepted synthetic pseudo rows in train; validation has 5 real plus 1 synthetic pseudo row; test has 5 real rows. Teacher representation: 633 weighted plus 32 style rows (665). Real train rows carry inferred duration scores; cache F0 uses RMVPE. Pseudo rows have trust `0.20`; teacher rows use their recorded trust weights.

Last logged step: total `1.503340`, flow `1.488785`, pitch `0.018467`, teacher `0.090877`. This is loss descent, not proof of singing quality. Validation audio is rendered separately by evaluation scripts. GPU memory, wall-clock, and full validation loss were not captured by this compact run and are missing evidence.
