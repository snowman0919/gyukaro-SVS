# v0.6 training record

- Independent score supervision: 24 target-F0-independent PyIN/script/spectrogram/CTC-reviewed rows, plus 76 high-confidence reconstructed rows with trust weights; 100 rows total. Pseudo singing is excluded from real-GYU prosody targets.
- Shared identity: 249 Fish/MOSS pair rows from 102 unique semantic groups, with atomic train/validation/test splits 191/32/26. A trust-weighted Barlow cross-view objective replaced the collapsed cosine-only pilot. Held-out Fish/MOSS cosine is 0.94029; nearest-centroid teacher leakage is 0.50000 and language clustering is 0.42308.
- Latent SoulX adapter: compact gated FiLM at SoulX `gt_decoder_inp`; SoulX backbone frozen. Spectral v0.5 adapter remains baseline.
- Targeted pseudo singing: 200 ACE-Step candidates generated from measured gaps; 45 passed RMVPE, duration, WavLM, ASR/LID, and degeneration gates at trust 0.20. It is generic low-trust transition evidence, not real-GYU prosody supervision.
- Prosody: v0.6 checkpoint trains the 24+76 score corpus, but independent evaluation does not show a consistent gain over v0.5. The production v0.6 renderer therefore retains the v0.5 prosody controller; v0.6 prosody remains an explicit experimental baseline.
