# Component traceability

| Concept | Source inspiration | Project problem | Implementation | Forward call | Loss / supervision | Evidence | Status |
|---|---|---|---|---|---|---|---|
| Unified phonemes | trilingual SVS frontend | opaque lyric text | `frontend/phonemizer.py:phonemize`, `UnifiedPhonemeEncoder` | `TriSingerModel.condition` | CFM; teacher representation | `test_all_hybrid_modules_receive_gradient` | integrated |
| Language features | language-aware singing frontend | Korean/English/Japanese timing differs | `LanguageFeatureEncoder` | `condition` added to content | CFM; teacher representation | KO/EN/JA frontend tests, teacher language gradient | integrated |
| Phoneme-note mapping | score-conditioned SVS | lyric cannot be whole-note TTS | `alignment/phrase.py:build_phrase_frames` | `ScoreEncoder` receives mapping index, onset, duration | CFM | alignment test, multi-note phrase render | integrated |
| Blurred boundary | TCSinger 2 | hard transition discontinuity | `BlurredBoundaryEncoder` | `condition` after content+score | CFM | boundary condition/gradient test; lower energy jump in current evaluation | integrated, quality not proven |
| Timbre | target-speaker adaptation | retain GYU reference identity | `TimbreEncoder` | broadcast into condition | weighted teacher representation | teacher batch timbre gradient; WavLM metric | integrated |
| Style/technique | TechSinger | preset and curve controls absent | `StyleEncoder` | broadcast into condition | CFM; teacher style rows | full-module gradient test | integrated, controls uncalibrated |
| Pitch curve | score SVS / FM-Singer conditioning | uncontrolled pitch | `PitchConditionEncoder` | added to condition | voiced masked log-F0 loss | pitch curve test; SoulX RMVPE evaluator | integrated, poor generated F0 |
| Conditional flow matching | FM-Singer | random-init waveform generation impractical | `ConditionalFlowTransformer` | predicts latent velocity in `forward` and Euler `sample` | rectified-flow MSE | flow objective and gradient tests | integrated |
| Codec acoustic decoder | pretrained codec reuse | tiny corpus cannot train waveform decoder | `inference/codec.py:MossCodecDecoder` | after sampled latent | frozen, no acoustic training loss | package smoke render | integrated |

`tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` observes non-zero gradients for every learned component. `test_teacher_distillation_reaches_timbre_and_language_encoders` repeats this for teacher loss. Connectivity evidence is not quality evidence.
