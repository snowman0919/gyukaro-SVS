# Training v0.2 report

Run: `PYTHONPATH=src python scripts/train_hybrid.py --steps 160` on NVIDIA GB10. Model: 96-D, three-layer phrase transformer, 3.0 MB checkpoint `checkpoints/gyu_hybrid_v0.2.pt`. Data: 60 real train / 5 validation / 5 test; 633 teacher representation rows. Final step total loss 2.353837, flow 2.289662, pitch 0.057003, teacher 0.408833. No accepted pseudo singing was used.
