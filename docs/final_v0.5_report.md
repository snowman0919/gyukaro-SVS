Overall status: INCOMPLETE — real Korean path runs, but held-out prosody quality gate is not met.
Current version: gyu-singer-v0.5-incomplete
Package: gyu-singer-v0.5-incomplete
Package SHA-256: `4ebd38c589427c96a8388024fccc80e7a0c39ecca5f182cb4ff5bf8c142f01f8`
Git commit: `6985eac`
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
Korean: end-to-end v0.5 WAV succeeds; prosody gate incomplete
English: experimental content path only
Japanese: experimental content path only

## Evidence

- Score reconstruction: `data/manifests/real_score_{candidates,accepted}.jsonl`, `docs/score_reconstruction_report.md`.
- Leakage: `data/manifests/real_gyu_prosody.jsonl` declares target-only F0; `tests/test_v05_semantics.py` fixes condition tensors while targets change.
- Alignment: `docs/phoneme_alignment_report.md` (mean vowel nucleus share 0.721; 76/76 above 50%).
- Teacher extraction: `docs/teacher_representation_distillation.md`; Fish shape 1024, MOSS shape 768, shared 32-D projection.
- Acoustic path: `docs/acoustic_style_adapter.md`; `artifacts/samples/gyu_v05_ko.wav`.
- Held-out prosody: `artifacts/reports/evaluation_v0.5_prosody.json` (mean correlation 0.2316; median pitch MAE 199.73 cents; nominal baseline 197.27 cents). Residual model does not yet improve the held-out target.

The package is deliberately marked incomplete. A WAV, F0 passthrough, gradient test,
or style-vector change is not treated as success. Next Goal: improve real-GYU
prosody target quality/model capacity and rerun held-out evaluation before any
experimental package claim.
