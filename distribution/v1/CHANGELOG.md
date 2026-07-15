# Changelog

## v1.0.0-rc.5

- Fix all-frame voiced F0 using canonical phoneme/note voicing; silence and unvoiced consonants now receive F0=0.
- Add MMS CTC latent timing correction for rapid Korean and English content.
- Select FP32 SoulX decode settings from identical-condition stress sweeps.
- Record human PASS for nine stress phrases and matched RC4 comparisons.
- Preserve `gyu-singer-v0.8` and tag `v1.0.0-rc.4` as failed audio-quality baseline.

## v1.0.0 release preparation

- Automated pinned model, runtime, and OpenUtau installation.
- Native multi-note OpenUtau phrase renderer with authoritative editor pitch and expressions.
- Resident health/model endpoints, serialized concurrent requests, failure recovery, and clean restart/shutdown.
- Real 2-minute, 136-note KO/EN/JA OpenUtau export and boundary-quality gate.
- Exact phrase cache invalidation/reuse tests and clean-package validation path.

## v0.9

- Initial maintained OpenUtau fork overlay and resident GYU v0.8 backend.

Earlier v0.1-v0.8 stages remain documented in the source repository as explicit research baselines.
