# GYU acoustic adaptation

The GYU stage trained 4,968 parameters for 600 steps with 14 real-GYU primary pairs and 30 public replay pairs. Validation used 5 held-out GYU rows and reached loss 2.518539.

This adapter improved several pair-reconstruction metrics, but did not beat the universal backbone on the actual nine-file production path. The RC6 runtime therefore selects the universal checkpoint at 25% residual strength. Singing and GYU adapters remain explicit measured baselines, not production components.

Identity preservation at the selected strength: WavLM before/after cosine 0.998206, ECAPA 0.997823; similarity-to-GYU deltas were -0.00064 and -0.001876. These are diagnostics, not listening evidence.

The spectral GYU follow-up trained a separate 6,576-parameter adapter on 14 real-GYU pairs with 187 deduplicated public replay rows; five real-GYU rows were held out. It improves held-out GYU spectral convergence at full strength (0.4455→0.3906), but underperforms the singing adapter on log-spectral and high-band reconstruction. At 50% on the nine RC6 files it also reduces HF spikes less than the singing adapter (375.93 versus 344.18) without a better interval result. It is rejected and not included in the listening candidate.
