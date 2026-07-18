# SoulX truncated identity supervision diagnostic design

## Purpose and boundary

Test whether the existing v0.7 identity FiLM adapter can improve final SoulX speaker identity when supervised on decoded audio instead of a latent centroid. This is a bounded diagnostic. It does not change the production renderer, RC8 policy, SoulX checkpoint, vocoder checkpoint, style adapter, packaging, or OpenUtau integration. A failed experiment is deleted from the runtime path; a passing experiment creates only a human A/B candidate.

The final decode remains one phrase-level 64-step SoulX decode. There is no per-note TTS, final-WAV stitching, waveform pitch shifting, or replacement foundation model.

## Differentiable path

Reuse the existing SoulX reverse-diffusion equations and `SoulXRealLatentAdapters.identity` module. For each training item and seed:

1. Run the first `64 - K` SoulX steps with gradients disabled, using the current adapter condition so the state matches the candidate being trained.
2. Detach that diffusion state.
3. Recompute the adapter-conditioned SoulX input with gradients enabled and run the final `K` steps through the frozen SoulX backbone.
4. Decode through the frozen vocoder with autograd enabled.
5. Backpropagate only into the identity FiLM adapter.

Run separate `K=2` and `K=4` experiments. SoulX and vocoder parameters have `requires_grad=False`; their operations remain in the graph only where required to carry gradients to the adapter. Style conditioning is disabled. The first feasibility check must prove finite, nonzero adapter gradients and record peak memory before any optimizer loop runs.

The candidate starts from the current v0.7 identity-adapter state. Before training, an identical seed/input render must match the current adapter-ON production path within floating-point tolerance. The identity-OFF production WAV is cached as the preservation target. Diffusion noise, source, score F0, reference, precision, CFG, and seed are identical between each OFF baseline and candidate.

## Data isolation

Use only existing phrase specifications and locally cached models. The fixed phrase split is:

- train: `examples/korean.json`, `examples/english.json`, `examples/japanese.json`;
- validation: `examples/quality_ko.json`, `examples/quality_en.json`, `examples/quality_ja.json`;
- held-out promotion: `examples/heldout_ko.json`, `examples/heldout_en.json`, `examples/review_sustain_ko.json`, `examples/review_large_interval_ko.json`, and `examples/review_phrase_boundary_ko.json`;
- protected regression only: `examples/review_rapid_ko.json`.

Freeze this manifest before optimization; no text/source/score combination may cross splits. Seeds `7`, `21`, and `42` are all represented during evaluation and no seed is selected after seeing results. This deliberately small training split tests transfer rather than creating a new corpus.

`examples/heldout_ja.json` is excluded from identity training, validation, and promotion because its OmniVoice source already contains lexical repetition collapse. It remains labeled as a separate content-source failure. Normal Japanese phrases whose source Whisper transcript passes the existing content gate may be used, but their split is frozen before training. Rapid KO is evaluation-only and remains a protected non-regression case.

## Objective

The loss is a weighted sum of:

- frozen WavLM cosine distance to the multi-reference GYU centroid;
- frozen ECAPA cosine distance to the same independently fixed GYU references;
- waveform L1 distance to the aligned identity-OFF production WAV;
- log-magnitude multi-resolution STFT distance at FFT sizes `256`, `1024`, and `4096`;
- frozen Whisper/content-feature distance to the identity-OFF WAV;
- differentiable voiced-frame pitch-period preservation against the identity-OFF WAV;
- identity-adapter output-energy and gate regularization.

GYU references and all loss weights are fixed before held-out evaluation. The existing reference audit remains authoritative; references are not selected per phrase. RMVPE is the final F0 judge rather than the differentiable training proxy.

Use the smallest adapter update that produces a measurable identity gain. Stop immediately on non-finite loss, zero gradient, out-of-memory, loss-path detachment, or baseline-initialization mismatch. Do not add a surrogate model or a new dependency.

## Selection and evaluation

Compare four fixed conditions under identical inputs:

1. identity OFF production baseline;
2. current v0.7 identity adapter;
3. truncated-backprop candidate with `K=2`;
4. truncated-backprop candidate with `K=4`.

Evaluate every held-out phrase and every seed `7/21/42`; report the full sample-wise distribution, not only means. Every generated WAV is checked directly with free Whisper, RMVPE, voicing accuracy, HF-spike ratio, sample-jump p99.9, clipping, WavLM, ECAPA, waveform plots, and FFT-256/1024/4096 plots. Record runtime and peak memory for both truncated variants.

A candidate may become a human A/B candidate only when all of these hold against the identity-OFF baseline:

- held-out mean WavLM-to-GYU improves by at least `0.01`;
- held-out mean ECAPA-to-GYU improves by at least `0.01`;
- no phrase/seed loses more than `0.02` WavLM or `0.03` ECAPA similarity;
- Whisper lyric similarity loses no more than `0.02` absolute and introduces no repeated or omitted phrase;
- RMVPE pitch MAE increases by no more than the larger of `2 cents` or `5%`;
- voicing accuracy decreases by no more than `0.01` absolute;
- HF-spike ratio and sample-jump p99.9 each increase by no more than `10%`;
- no clipping is introduced;
- Rapid KO passes the same content, pitch, voicing, and artifact limits.

When both `K=2` and `K=4` pass, select the smaller `K` unless `K=4` improves both speaker metrics by at least another `0.01` without weakening any preservation metric. This keeps the experiment and any later integration minimal.

## Failure and promotion policy

Any failed mandatory gate rejects that checkpoint. Rejected checkpoints are not referenced by a renderer or package, and existing production output hashes remain unchanged. The report keeps the failure evidence and labels the result `diagnostic_reject`.

Passing the automated gate only creates paired listening WAVs and a `human_pending` report. It does not connect the adapter to RC8. Runtime integration requires a later explicit human-listening pass and a separate runtime change with dataset validation and package smoke testing.

## Reproducible evidence

The diagnostic writes a frozen manifest, training history, exact model revisions, loss weights, seed list, peak-memory/runtime measurements, sample-wise metrics, actual WAV paths, and waveform/multi-resolution-STFT comparison paths under one report directory. Generated audio and local model caches remain uncommitted. Source recordings under `data/source/` are never modified or committed.
