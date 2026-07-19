# Korean Phone-Centered Foundation Reassessment

## Conclusion

`foundation_machine_inconclusive`

The experiment did not qualify the GTSinger soprano foundation for Korean linguistic adaptation. Training remains blocked and no identity work was performed.

## Protocol and evidence

The frozen probe includes ordinary, rapid, sustained, large-interval, phrase-boundary, single-syllable, repeated-syllable, coda, tense, aspirated, liaison, nasalization, and liquid-assimilation cases. Existing foundation renders cover five stress cases across seeds 7, 21, and 42, for 15 evaluated WAVs.

Five script-known real GYU references calibrated MMS target-conditioned alignment. Matched mean log scores averaged -1.340109; deliberately mismatched scripts averaged -5.330454. The mean separation was 3.990345 and the minimum per-reference margin was 3.557688. This demonstrates that the aligner reacts to script mismatch, but it does not turn forced alignment into independent phone recognition.

Candidate mean alignment scores were:

- ordinary: -3.524283
- rapid: -3.668777
- large interval: -3.385465
- sustain: -1.690442
- phrase boundary: -4.592442

Cross-seed HuBERT content consistency was 0.999795 across 15 outputs. This is strong seed stability, not proof that the intended Korean phones were produced.

Only `ko_components_v1` has rendered evidence. The canonical and onset-rhyme candidates were encoded and coverage-checked but not rendered. Therefore `representation_selection` is null and no fair representation comparison exists.

## Whisper boundary

Whisper transcripts are retained only as auxiliary observations with primary weight 0. They cannot change the phone-centered decision.

## Reproducibility

- Frozen probe: `configs/korean_phone_probe.json`
- Evaluator: `scripts/evaluate_korean_phone_reassessment.py`
- Compact evaluation: `artifacts/reports/korean_phone_reassessment/evaluation.json`
- Alignment audit: `data/manifests/korean_alignment_audit.jsonl`
- Local ignored listening evidence: `data/external/work/korean_phone_reassessment_review/`

The final decision is inconclusive because no calibrated independent local Korean singing phone recognizer was available. Defining a threshold after seeing these outputs would invalidate the experiment.
