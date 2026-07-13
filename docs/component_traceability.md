# Component traceability

| Component | Forward path | Objective/evidence |
|---|---|---|
| UnifiedPhonemeEncoder + LanguageFeatureEncoder | `TriSingerModel.condition` | multilingual frontend test |
| ScoreEncoder | condition | phrase MIDI/note frame test |
| BlurredBoundaryEncoder | condition | note boundaries soften 5-frame context |
| TimbreEncoder | condition | real reference acoustic summary |
| StyleEncoder | condition | five scalar controls |
| PitchConditionEncoder | condition | masked log-F0 loss |
| ConditionalFlowTransformer | `forward` / `sample` | flow-matching loss |
| SingingDecoder | render latent | acoustic-bias loss term |
| Frozen MOSS codec | `HybridRenderer.render` | actual clean-package WAV smoke |

`tests/test_hybrid.py::test_all_hybrid_modules_receive_gradient` verifies nonzero gradients for every learned component.
