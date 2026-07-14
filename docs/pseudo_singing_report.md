# Pseudo-singing report

Three Fish S2 `[singing]` then SoulX-Singer SVC probes were measured: KO RMVPE 0.9918/WavLM 0.8585/ECAPA 0.7218, EN 0.9778/0.9632/0.6769, JA 0.9964/0.8812/0.7253. Candidate manifests preserve their provenance, inferred-score label, 48 kHz output, metric results, and `quality_status: rejected_license`.

Fish Audio Research License prohibits using its outputs to improve a foundational generative AI model. `scripts/admit_pseudo_probes.py` therefore gives these rows `trust_weight: 0.0`, `training_license: prohibited_for_foundational_generative_ai_training`, and `training_use: evaluation_only_not_training`; accepted manifest is intentionally empty. `prepare_hybrid_data.py` requires both `quality_status: accepted` and `training_license: allowed` before an acoustic target can enter training.

No legally admitted pseudo-singing row exists yet. Target remains 100–500 diverse candidates from a pipeline whose teacher/output license explicitly allows this use. SoulX direct SVS is being evaluated as the replacement; its isolated preprocessing environment is not yet dependency-complete.
