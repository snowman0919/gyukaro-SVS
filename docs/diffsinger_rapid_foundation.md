# DiffSinger rapid-source qualification

Status: objective source gate passed at step 15,000; human listening pending. No release is authorized by this report.

## Why the PJS source was rejected

The external 1.82-second Japanese stress phrase contains 44 phones and four lyric repetitions. The known-correct source itself scores only `0.0741` with free Whisper similarity, so free ASR is not a valid lexical oracle for this phrase. Qualification instead compares teacher-forced lyric NLL with the correct source (`2.4902`, maximum accepted `2.6147`) and also requires F0 p90 error at most 100 cents, gross pitch error at most 5%, voiced ratio at least 0.8, and no clipping.

The best explicitly voiced PJS checkpoint failed: teacher-forced NLL `2.7090`, F0 p90 error `379.15` cents, gross pitch error `10.14%`, and voiced ratio `0.75`. Common Voice speaker-separated transfer, final-diffusion training, joint decoder training, gain/depth/chunk sweeps, same-singer paired speech, and neutral pitch/rate augmentation also failed. The PJS path is therefore not eligible for GYU adaptation or OpenUtau packaging.

Two causal defects were fixed rather than hidden:

- pitch/rate augmented examples no longer select shifted formant or compressed-rate speaker controls;
- silence, Japanese obstruents, and explicitly devoiced vowels receive zero F0 during both training and inference.

These corrections did not make the 100-song foundation sufficiently capable, so thresholds were not weakened.

## Replacement foundation

The next bounded candidate uses the pinned neutral Control_Group subset of the GTSinger Japanese soprano corpus: 563 score-aligned singing segments, 1.79 hours total, including 307 fast segments. The corpus provides manual phoneme alignment and realistic MusicXML. Technique groups are excluded from the neutral lexical foundation instead of being averaged into one speaker ID. Complete songs are held out so duplicate performances cannot cross train and validation splits. The acoustic model remains OpenVPI DiffSinger; this is a data/capacity correction, not a new runtime architecture.

GTSinger is CC BY-NC-SA 4.0. Any checkpoint or package derived from it must be attributed, non-commercial, and share-alike. The dataset audio stays under `data/external/` and is excluded from Git and release archives.

## Measured result

The 15,000-step auxiliary checkpoint passes the external 1.82-second rapid gate:

- teacher-forced lyric NLL: `2.5059` (maximum `2.6147`);
- F0 median / p90 absolute error: `4.33` / `35.11` cents;
- gross pitch error over 600 cents: `0.00%`;
- observed voiced ratio: `0.8913`;
- clipping fraction: `0.0`.

Step 13,000 also passes, so the result is not isolated to one checkpoint. The authoritative machine-readable result is `artifacts/reports/diffsinger_gtsinger_ja_source_evaluation_11k_15k.json`.

The original evaluator stretched a 91-frame target across a 92-frame render and therefore interpolated nonzero pitch through zero-F0 consonant boundaries. This produced false octave errors. The corrected evaluator compares the equal 20 ms grids directly, permits only one trailing-frame difference, and rejects larger timing drift. It never time-warps F0 and has regression tests for both cases.

This result authorizes the frozen-source final diffusion experiment. It does not authorize GYU adaptation or packaging until the generated rapid source receives human listening approval. The checkpoint remains a CC BY-NC-SA 4.0 derivative.

## Final diffusion probe

Only the 10,299,008-parameter diffusion path was trained for 2,000 updates; the qualified phoneme encoder and auxiliary decoder stayed frozen. Two final settings pass the same gate:

| setting | lyric NLL | F0 p90 | gross error | voiced ratio | clipping |
|---|---:|---:|---:|---:|---:|
| depth 0.4, 20 steps | 2.6113 | 24.55 cents | 0% | 0.9457 | 0 |
| depth 0.6, 50 steps | 2.5605 | 27.40 cents | 0% | 0.9891 | 0 |

Depth 0.4 / 20 steps is the objective default because it preserves more unvoiced structure and is cheaper. Human comparison against depth 0.6 remains mandatory; neither setting authorizes GYU adaptation, OpenUtau packaging, or release yet. Evidence: `artifacts/reports/diffsinger_gtsinger_ja_diffusion_evaluation.json`.
