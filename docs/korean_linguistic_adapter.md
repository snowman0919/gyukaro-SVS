# Korean Linguistic Adapter Diagnostic

## Status

`blocked_no_selected_representation`

Optimizer steps: 0. No checkpoint was trained or selected.

## Bounded architecture

The adapter embeds the selected Korean phone representation and projects it into the frozen foundation hidden size through a bounded residual. Its final projection is initialized to zero, so the initialized adapter is numerically equivalent to the unadapted foundation.

The acoustic decoder, variance predictors, pitch predictor, duration predictor, vocoder, and soprano speaker condition are frozen. Only Korean phone embeddings and the linguistic projection are eligible to train. The configuration contains no speaker-identity objective.

## Data trust

`diffsinger_ko_phoneme_prior.jsonl` is low-trust teacher-speech linguistic evidence. `real_phoneme_alignment_all.jsonl` is inferred real-GYU alignment evidence with reduced weight. Neither source is relabeled as manually verified.

The combined audit contains 149 rows: 149 inferred-only, 0 manually verified. Because the lexical foundation gate is inconclusive and no phone representation was selected, feasibility does not authorize optimizer initialization.

Configuration: `configs/korean_linguistic_adapter.json`.
