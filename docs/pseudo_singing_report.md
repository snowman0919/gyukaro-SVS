# Pseudo-singing report

Three Fish S2 `[singing]` then SoulX-Singer SVC multilingual probes passed admission: KO RMVPE 0.9918/WavLM 0.8585/ECAPA 0.7218, EN 0.9778/0.9632/0.6769, JA 0.9964/0.8812/0.7253. `scripts/admit_pseudo_probes.py` writes provenance, metric gates, inferred-score label, 48 kHz output, `quality_status: accepted`, and trust `0.20` to both required manifests.

`scripts/prepare_hybrid_data.py` includes all three accepted rows: EN and JA enter train, KO is deterministic validation holdout. The 1,200-step checkpoint therefore uses synthetic pseudo-singing acoustic targets in its real flow-loss path. Their scores are explicitly `inferred_from_RMVPE_pilot_contour_not_manual_score`; do not treat them as verified score labels.

This is a 3-item multilingual pilot, not the requested 100–500 corpus. Register/phrase coverage is inadequate for quality claims. Gate dimensions recorded per candidate: RMVPE F0 correlation, duration ratio, Whisper content score, WavLM/ECAPA speaker similarity, language, and output WAV provenance. No simple autocorrelation F0 gate is used.
