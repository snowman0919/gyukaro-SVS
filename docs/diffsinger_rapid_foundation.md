# DiffSinger rapid-source qualification

Status: rejected. The earlier objective-pass claim was invalid, and no release is authorized.

## Why the PJS source was rejected

The external 1.82-second Japanese stress phrase contains 44 phones and four lyric repetitions. The earlier gate incorrectly substituted teacher-forced lyric NLL when free Whisper failed. This let acoustically unintelligible output pass. Teacher-forced NLL is now diagnostic only: every candidate must produce a matching free transcript and pass waveform/F0 checks.

The comparison inputs were also invalid. The candidate duration is `1.8228` seconds while the alleged reference is `2.3499` seconds. Their F0 distributions do not describe the same signal: the score target has median `666.66 Hz` (MIDI `76.19`), while the reference RMVPE track has median `118.94 Hz` (MIDI `46.35`). The target therefore explains the user's excessive-pitch finding but cannot validate similarity to that reference.

The best explicitly voiced PJS checkpoint failed: teacher-forced NLL `2.7090`, F0 p90 error `379.15` cents, gross pitch error `10.14%`, and voiced ratio `0.75`. Common Voice speaker-separated transfer, final-diffusion training, joint decoder training, gain/depth/chunk sweeps, same-singer paired speech, and neutral pitch/rate augmentation also failed. The PJS path is therefore not eligible for GYU adaptation or OpenUtau packaging.

Two causal defects were fixed rather than hidden:

- pitch/rate augmented examples no longer select shifted formant or compressed-rate speaker controls;
- silence, Japanese obstruents, and explicitly devoiced vowels receive zero F0 during both training and inference.

These corrections did not make the 100-song foundation sufficiently capable, so thresholds were not weakened.

## Replacement foundation

The next bounded candidate uses the pinned neutral Control_Group subset of the GTSinger Japanese soprano corpus: 563 score-aligned singing segments, 1.79 hours total, including 307 fast segments. The corpus provides manual phoneme alignment and realistic MusicXML. Technique groups are excluded from the neutral lexical foundation instead of being averaged into one speaker ID. Complete songs are held out so duplicate performances cannot cross train and validation splits. The acoustic model remains OpenVPI DiffSinger; this is a data/capacity correction, not a new runtime architecture.

GTSinger is CC BY-NC-SA 4.0. Any checkpoint or package derived from it must be attributed, non-commercial, and share-alike. The dataset audio stays under `data/external/` and is excluded from Git and release archives.

## Corrected measured result

The 15,000-step auxiliary checkpoint is rejected:

- free Whisper transcript does not match the requested lyric (`0.0690` similarity);
- teacher-forced lyric NLL `2.5059` is retained only as a diagnostic;
- F0 median / p90 absolute error: `4.33` / `35.11` cents;
- gross pitch error over 600 cents: `0.00%`;
- observed voiced ratio: `0.8913`;
- clipping fraction: `0.0`.

No evaluated source checkpoint passes. The authoritative machine-readable result is `artifacts/reports/diffsinger_gtsinger_ja_source_evaluation_11k_15k.json`.

The original evaluator stretched a 91-frame target across a 92-frame render and therefore interpolated nonzero pitch through zero-F0 consonant boundaries. This produced false octave errors. The corrected evaluator compares the equal 20 ms grids directly, permits only one trailing-frame difference, and rejects larger timing drift. It never time-warps F0 and has regression tests for both cases.

This result does not authorize final diffusion, GYU adaptation, OpenUtau packaging, or release. The checkpoint remains a CC BY-NC-SA 4.0 derivative.

## Final diffusion probe

Only the 10,299,008-parameter diffusion path was trained for 2,000 updates; the phoneme encoder and auxiliary decoder stayed frozen. Both previously reported settings are rejected:

| setting | free Whisper transcript | similarity | F0 median | voiced ratio |
|---|---|---:|---:|---:|
| depth 0.4, 20 steps | unrelated closing phrase | 0.1176 | 672.19 Hz | 0.9457 |
| depth 0.6, 50 steps | unrelated phrase | 0.0645 | 672.68 Hz | 0.9891 |

Neither is a default or adaptation input. Human listening also rejected both as excessively high and unintelligible. Evidence: `artifacts/reports/diffsinger_gtsinger_ja_diffusion_evaluation.json`.
