# GYU acoustic adaptation

The GYU stage trained 4,968 parameters for 600 steps with 14 real-GYU primary pairs and 30 public replay pairs. Validation used 5 held-out GYU rows and reached loss 2.518539.

This adapter improved several pair-reconstruction metrics, but did not beat the universal backbone on the actual nine-file production path. The RC6 runtime therefore selects the universal checkpoint at 25% residual strength. Singing and GYU adapters remain explicit measured baselines, not production components.

Identity preservation at the selected strength: WavLM before/after cosine 0.998206, ECAPA 0.997823; similarity-to-GYU deltas were -0.00064 and -0.001876. These are diagnostics, not listening evidence.
