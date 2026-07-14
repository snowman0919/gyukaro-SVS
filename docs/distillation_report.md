# Teacher distillation report

Input manifests are `teacher_weighted.jsonl` (633) and `teacher_style_supplement_weighted.jsonl` (32): 665 rows total. `scripts/train_hybrid.py` reads both, selects one deterministic weighted row per training step, passes teacher audio only through `acoustic_reference_features`, and applies `weighted_distillation_loss(model.distillation_prediction(...), teacher_features, trust_weight)`.

Teacher rows never load codec acoustic targets and never supervise `SingingDecoder` as GYU singing truth. `trust_weight` is multiplied in the actual loss; zero trust gives zero contribution (`test_losses_use_pitch_mask_and_teacher_trust`). The teacher branch backward test verifies nonzero `TimbreEncoder` and `LanguageFeatureEncoder` gradients.

Current run: teacher loss coefficient `0.15`; one teacher representation update per acoustic update. Stage plan is A teacher representation, B pseudo acoustic, C real-anchor adaptation, D joint low-LR refinement. Actual 1,200-step run is joint C/D bootstrap: 60 real and 2 low-trust pseudo train rows round-robin with teacher representation loss. It is not a completed staged training programme.

Compatible 80-step A/B evidence is `artifacts/reports/hybrid_teacher_ablation.json`. WavLM cosine to GYU reference: no-teacher KO/EN/JA `0.5288/0.5582/0.5165`; weighted teacher `0.5310/0.5429/0.5169`. Teacher changes outputs, but EN declines and this short run does not establish a quality gain.
