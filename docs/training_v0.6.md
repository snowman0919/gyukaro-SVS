# v0.6 training record

- Independent score manifest: 24 rows, PyIN plus script-pattern/spectrogram review, RMVPE target excluded from score construction.
- Shared identity: 191 benchmark-ID-disjoint Fish/MOSS pairs, 64-dimensional space, weighted cross-view cosine agreement; no fabricated speaker negatives.
- Latent SoulX adapter: 296,193 trainable parameters in a compact gated FiLM module; SoulX backbone frozen. The current checkpoint is calibration-only and requires stronger style supervision before production claims.
- Prosody: v0.5 real-GYU controller retained; independent evaluation does not show a reliable aggregate gain over nominal score.
