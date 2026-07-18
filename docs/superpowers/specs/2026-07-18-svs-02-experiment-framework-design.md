# SVS-02 Korean Experiment Framework Design

## Status

This document defines a diagnostic experiment framework. It does not approve a model for production, packaging, or OpenUtau.

## Question

Can the frozen GTSinger soprano foundation render Korean phonetic distinctions reliably enough to justify a bounded linguistic adapter experiment?

## Frozen protocol

- Probe cases: `configs/korean_phone_probe.json`
- Seeds: 7, 21, 42
- Candidate representations: `ko_components_v1`, `ko_canonical_v1`, `ko_onset_rhyme_v1`
- Primary evidence: phone-centered alignment and content stability
- Whisper: auxiliary observation only, primary weight 0
- Foundation: frozen GTSinger Japanese soprano checkpoint
- Identity objective: prohibited

Representation A preserves explicit Hangul onset, nucleus, and coda components. Representation B maps the same distinctions to canonical Korean phone symbols without merging tense and aspirated consonants. Representation C groups onset and rhyme while retaining an explicit no-coda state. The experiment may select a representation only after comparable rendered evidence exists for all candidates.

## Evidence boundary

The local MMS forced aligner is target-conditioned. It can show whether a monotonic target path exists and distinguish matched from deliberately mismatched scripts, but it is not an independent Korean singing phone recognizer. HuBERT cross-seed similarity measures stability, not correctness. Neither measurement alone establishes lexical validity.

Consequently, missing calibrated independent recognition produces `foundation_machine_inconclusive`, not a pass. No threshold may be invented after candidate output is observed.

## Adapter boundary

The zero-initialized `KoreanLinguisticAdapter` is a feasibility scaffold. If and only if a representation is selected, the acoustic decoder, variance, pitch and duration predictors, vocoder, and soprano conditioning remain frozen. Only Korean phone embeddings and the linguistic projection may train. Identity loss is excluded.

## Decisions

- A foundation rejection stops adapter training.
- An inconclusive lexical gate also stops adapter training.
- Whisper cannot override the phone-centered decision.
- Human review may inspect evidence but cannot be labeled a promotion candidate while the machine gate is inconclusive.
