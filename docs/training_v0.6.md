# v0.6 training record

- Independent score manifest: 24 rows, PyIN plus script-pattern/spectrogram review, RMVPE target excluded from score construction.
- Shared identity: 191 benchmark-ID-disjoint Fish/MOSS pairs, 64-dimensional space, weighted cross-view cosine agreement; no fabricated speaker negatives.
- Latent SoulX adapter: 296,193 trainable parameters in a compact gated FiLM module; SoulX backbone frozen. The current checkpoint is calibration-only and requires stronger style supervision before production claims.
- Prosody: v0.6 checkpoint trains 24 verified + 76 high-confidence reconstructed rows with trust labels; it is used by the v0.6 renderer. Independent evaluation shows only a small MAE change and no decisive correlation gain.
