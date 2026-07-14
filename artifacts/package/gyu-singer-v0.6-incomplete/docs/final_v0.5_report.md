Overall status: EXPERIMENTAL
Current version: gyu-singer-v0.5-experimental
Package: gyu-singer-v0.5-experimental
Package SHA-256: `9943544a58e360ea6a37c6e61d7f6451047f88a3678e61909f22c2c702e1b4cc`
Git commit: `9345cdb`
Real score reconstruction: active, inferred RMVPE + script-constrained candidates/accepted
Deprecated fake-score path used in v0.5 training: no
Target F0 leakage removed: yes; production condition excludes target F0
Real GYU prosody supervision: trained on 76 accepted inferred-score rows
Singing phoneme alignment: MMS CTC + Korean vowel-nucleus prior
Acoustic-style adapter in final audio path: yes, pre-SoulX spectral representation
Teacher-internal representation distillation: Fish DAC + MOSS tokenizer paths integrated
Teacher models with internal representations used: Fish S2 Pro, MOSS Local v1.5 tokenizer
Pseudo singing used: historical 27 accepted rows; no blind v0.5 expansion
Phrase-level generation: OmniVoice + SoulX-Singer
v0.4 fallback used by v0.5 backend: no
OpenUtau readiness: protocol v2 bridge preserved; native phrase renderer deferred
Korean: end-to-end v0.5 WAV succeeds; held-out prosody gate passes exploratory threshold
English: experimental content path only
Japanese: experimental content path only

## Evidence

- Score reconstruction: `data/manifests/real_score_{candidates,accepted}.jsonl`, `docs/score_reconstruction_report.md`.
- Leakage: `data/manifests/real_gyu_prosody.jsonl` declares target-only F0; `tests/test_v05_semantics.py` fixes condition tensors while targets change.
- Alignment: `docs/phoneme_alignment_report.md` (mean vowel nucleus share 0.721; 76/76 above 50%).
- Teacher extraction: `docs/teacher_representation_distillation.md`; Fish shape 1024, MOSS shape 768, shared 32-D projection.
- Acoustic path: `docs/acoustic_style_adapter.md`; `artifacts/samples/gyu_v05_ko.wav`.
- Final-audio style probe: `artifacts/reports/acoustic_style_evaluation_v0.5.json` (same phrase/score, five style presets; spectral centroid and RMS change).
- Held-out prosody: `artifacts/reports/evaluation_v0.5_prosody.json` (mean correlation 0.8667; median pitch MAE 53.16 cents; nominal baseline 57.52 cents; +3.80 cents median improvement).

A WAV, F0 passthrough, gradient test, or style-vector change alone is not treated
as success. This experimental package is justified by held-out real-GYU F0
evaluation, final-audio style probes, distinct backend identity, and clean package
render. Next Goal: expand held-out Korean coverage and calibrate EN/JA.
