Overall status: RC6 objective candidate; HUMAN LISTENING PENDING; final release blocked
Current version: 1.0.0-rc.6-candidate
Package: artifacts/package/gyu-singer-v1.0.0-rc6-candidate.zip
Package SHA-256: 041adbd9d26def69335001b64a4404313e3b5ce4dc064d9cd515f62a2eda1dde
Git commit: 780a6e5bda42ac4d4167e38957c1bf8ac96e6ce0
Clean package install: PASS
Primary artifact source: RC4 content/score timing mismatch plus all-frame voiced F0
Secondary artifact sources: low-step/high-CFG decode and hard unedited note steps
Selected SoulX settings: FP32; standard 32/CFG1.5, rapid 64/CFG2.0, large interval 32/CFG2.0 seed 21
Selected acoustic refiner: universal backbone at 25%
Disabled components: singing adapter and GYU adapter (production regressions); BigVGAN diagnostic (mixed regressions)
Phrase-level generation: yes
Per-note TTS used: no
Waveform pitch shifting used: no
OpenUtau long-form: PASS (136 notes, 17 phrases, 119.982857 seconds)
OpenUtau edit behavior: PASS (note, lyric, PITD, style, cache invalidation)
Runtime stress: PASS
Korean: objective stress rendered; human pending
English: objective stress rendered; human pending
Japanese: objective stress rendered; human pending
Release recommendation: DO NOT tag or publish v1.0.0

# RC6 artifact-recovery report

RC4 is preserved exactly. RC5's canonical timing/voicing and safer SoulX policy fixed the identified structural defects. RC6 adds a small bounded, activity-gated residual refiner trained on actual SoulX reconstruction pairs. It does not denoise source recordings and does not replace SoulX.

Compared with RC5 across nine files, HF-spike ratio changed from 485.596389 to 458.558778, spectral flux from 0.232432 to 0.23186, sample jump from 0.096265 to 0.093596, voicing from 0.870611 to 0.872033, and pitch MAE from 8.47 to 8.467778 cents. ASR similarity is unchanged at 0.924211; clipping is zero.

Remaining audible defects are unknown until human review. Objective risks include a small aggregate HF-energy increase, persistent English “Sing”/“Sink” ambiguity, sustained-vowel ASR ambiguity, and weak voicing scores for large-interval and phrase-boundary cases.

Before/after files and the nine final candidate WAVs are in `artifacts/reports/rc6_listening_gate/`. The required next outcome is a per-file human verdict, not a final release.
