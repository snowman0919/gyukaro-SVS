# Production quality-path component traceability

`hybrid-soulx-phrase` loads `gyu_quality_pitch_controller.pt`, a
`TriSingerModel(latent_dim=1)`. Its UnifiedPhoneme, language, score, blurred
boundary, timbre, style, pitch, conditional-flow, and SingingDecoder modules
produce a bounded expressive F0 residual from score/control/reference inputs.
That residual is added to the nominal score F0 and is consumed by the frozen
SoulX neural acoustic decoder. The compact `hybrid-svs` codec-latent checkpoint
remains a failed experimental path; it is not the quality candidate.

| Concept | Source inspiration | Project problem | Implementation | Forward call | Loss / supervision | Evidence | Status |
|---|---|---|---|---|---|---|---|
| Unified phonemes | trilingual SVS frontend | opaque lyric text | `frontend/phonemizer.py:phonemize`, `UnifiedPhonemeEncoder` | `QualityPitchController.predict` -> `TriSingerModel.sample` | residual-flow F0 target; teacher representation in compact stage | gradient test; production score render | integrated |
| Language features | language-aware singing frontend | Korean/English/Japanese timing differs | `LanguageFeatureEncoder` | quality controller condition | residual-flow F0 target; teacher representation in compact stage | KO/EN/JA frontend tests, production score render | integrated |
| Phoneme-note mapping | score-conditioned SVS | lyric cannot be whole-note TTS | `alignment/phrase.py:build_phrase_frames`, `scripts/align_real_phonemes.py` | quality controller `ScoreEncoder` receives mapping index, onset, duration | residual-flow F0 target | MMS CTC plus singing-vowel prior; alignment regression test | integrated; real labels inferred |
| Blurred boundary | TCSinger 2 | hard transition discontinuity | `BlurredBoundaryEncoder` | quality controller condition before residual flow | residual-flow F0 target | boundary condition/gradient test; quality gate | integrated |
| Timbre | target-speaker adaptation | retain GYU reference identity | `TimbreEncoder` | GYU reference broadcast into quality controller condition | residual-flow F0 target; weighted teacher representation in compact stage | teacher batch timbre gradient; WavLM metric | integrated |
| Style/technique | TechSinger | preset and curve controls absent | `StyleEncoder` | quality controller condition; ACE phrase style prompt | residual-flow F0 target; teacher style rows in compact stage | full-module gradient test | integrated, controls partly calibrated |
| Pitch curve | score SVS / FM-Singer conditioning | actual-F0 leakage and uncontrolled pitch | `PitchConditionEncoder`, `QualityPitchController` | score nominal F0/control -> flow residual -> SoulX F0 input | synthetic quality-runtime RMVPE residual target | controller contour regression; resident quality gate | integrated |
| Conditional flow matching | FM-Singer | expressive contour from score is missing | `ConditionalFlowTransformer`, `SingingDecoder` | controller decoder produces residual source; flow refines it before SoulX decode | source plus residual-flow RMVPE-F0 MSE | flow/gradient tests; resident quality gate | integrated |
| Neural acoustic decoder | pretrained singing decoder reuse | tiny corpus cannot train waveform decoder | `SoulX-Singer SVC` worker | complete phrase content plus controller-conditioned explicit F0 | frozen Apache-2.0 decoder; controller F0 losses | package smoke and KO/EN/JA gate | integrated |

Real GYU score rows are reconstructed from RMVPE piecewise-note candidates plus recording-script priors by `scripts/reconstruct_real_scores.py`; `score_source` explicitly says inferred.  `tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` observes non-zero gradients for every learned component. `test_teacher_distillation_reaches_timbre_language_and_style_encoders` repeats this for teacher loss. Connectivity evidence is not quality evidence: the reconstructed-score compact retrain failed its quality gate (`artifacts/reports/hybrid_score_reconstructed_evaluation.json`).
