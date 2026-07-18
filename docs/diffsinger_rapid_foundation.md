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

## Bounded GYU LayerNorm adaptation probe

A final acoustic-projection-only adaptation was tested first and rejected: free-Whisper lyric similarity fell from `1.0` to `0.4` at every 100/200/300-step checkpoint, while WavLM-to-GYU also declined by step 300. This shows that globally rewriting the output projection damages Japanese lexical acoustics rather than safely transferring identity. Evidence: `artifacts/reports/diffsinger_gtsinger_gyu_acoustic_projection/evaluation.json`.

A later native DiffSinger probe adapted only speaker-conditioned LayerNorm affine parameters while replaying the Japanese foundation corpus. On the independent rapid C4 phrase, every 200/400/600-step render produced the exact free-Whisper transcript and kept RMVPE pitch p90 error near 36 cents with no clipping. It still failed the identity requirement: the best GYU-conditioned row changed WavLM-to-GYU only from `0.63353` to `0.63382`; ECAPA changed from `0.06300` to `0.07022`. This is not a meaningful GYU identity transfer, so the candidate is rejected despite its lexical and pitch accuracy. Evidence: `artifacts/reports/diffsinger_gtsinger_gyu_mixln/evaluation.json`.

This separates the two problems: the earlier depth-0.4/0.6 files were high and unintelligible because they combined a median-666 Hz score with an unqualified diffusion path; the corrected C4 foundation can be intelligible and score-accurate, but the available 4.742 minutes of inferred-label GYU phrase supervision is not sufficient to turn it into a verified GYU voice through this bounded adapter.

## Five-phrase held-out waveform and STT gate

The single rapid C4 phrase was not sufficient evidence. A new evaluation-only set uses five pinned GTSinger Japanese phrases whose source recordings all receive exact free-Whisper transcripts. Dataset-provided manual phoneme timing is preserved and the target F0 is independently re-extracted with RMVPE. This is GTSinger evidence, not GYU supervision. The builder and manifest are `scripts/build_diffsinger_gtsinger_heldout_set.py` and `artifacts/reports/diffsinger_gtsinger_heldout_set/manifest.json`.

Every rendered WAV was analyzed directly. Each report records SHA-256, free Whisper transcript, RMVPE pitch and voicing, clipping, spectral/HF discontinuity evidence, and WavLM/ECAPA similarity distributions against five real GYU references. The same score, seed, and depth-zero auxiliary decode were used for the soprano foundation, tenor transfer, and 20% GYU speaker mix.

| candidate | valid phrases | STT mean / minimum | max F0 p90 | min voicing accuracy | max HF/reference | WavLM / ECAPA mean |
|---|---:|---:|---:|---:|---:|---:|
| soprano foundation | 5/5 | 0.9244 / 0.8372 | 33.12 cents | 0.8808 | 1.070x | 0.54093 / 0.09710 |
| tenor transfer | 0/5 | 0.6298 / 0.0000 | 28.38 cents | 0.8160 | 2.901x | 0.74056 / 0.21269 |
| tenor + 20% GYU | 1/5 | 0.6742 / 0.4444 | 34.73 cents | 0.7943 | 3.023x | 0.73723 / 0.21152 |

The five-phrase validity rule is STT similarity at least `0.8`, pitch p90 at most `100` cents, gross pitch error at most `5%`, target-grid voicing accuracy at least `0.8` for the foundation or no more than `0.02` below its matching baseline for an adaptation, no clipping, HF spike at most twice the matching source, and no material sample-jump regression. One failed phrase rejects the entire candidate. The soprano source passes this machine gate but is not a GYU voice, so it is only a qualified foundation and not an RC. The tenor and GYU-mix candidates are rejected: their higher speaker similarity accompanies severe lexical collapse, voicing/sample-jump regressions, and reference-relative HF spikes. In the worst tenor row Whisper returns only repeated `ドゥー`; the mix row is also fragmented. A speaker score therefore cannot rescue either candidate.

The evaluator now compares target and observed voiced/unvoiced state on the same 20 ms F0 grid. It no longer treats an absolute observed voiced ratio as correctness for these longer phrases with rests and unvoiced regions. The raw observed ratios remain in every report for audit.

Authoritative reports:

- `artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_gtsja0165.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_gtsja0172.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_gtsja0174.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_gtsja0379.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/evaluation_gtsja0380.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/aggregate_evaluation.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/seed_stability/`

At fixed depth zero, seeds 7/21/42 produce different WAV hashes, so all fifteen soprano phrase×seed outputs were evaluated rather than assumed identical. Every one passes lexical, pitch, voicing, clipping, and HF gates. This establishes seed-stable score-native lexical/pitch behavior for the soprano foundation only. No human listening pass is claimed. The batch isolates the current failure to the tenor/GYU adaptation path rather than score timing or the qualified soprano acoustic foundation. It does not authorize identity adaptation, an acoustic refiner, RC8, OpenUtau packaging, or release.

## Identity-checkpoint freeze-contract defect

Checkpoint comparison found a concrete adaptation defect. The selected tenor-500 checkpoint had changed 137 tensors, including about 7.88 million encoder and 8.15 million auxiliary-decoder parameters. The earlier single rapid phrase happened to pass, but the five-phrase gate exposed the resulting lexical and HF instability.

The older soprano-based `gtsinger_ja_gyu_identity` checkpoint had a separate freeze-contract violation. Its report claimed that only the GYU speaker row and Korean text rows were trained, but step 100 also changed eight stretch-embedding/RNN tensors (about 918 thousand parameters) plus the shared AP/SP text rows. With GYU weight set to exactly zero, Whisper returned `ご視聴ありがとうございました` at `0.1081` similarity and the voiced ratio fell to `0.5439`. The failure therefore existed before speaker interpolation.

The remapped initial checkpoint was not the cause: under the combined dictionary its 100% soprano render was byte-for-byte identical to the qualified foundation. A strict checkpoint compositor now starts from that initial checkpoint and copies only `ko_*` text rows and GYU speaker row 1 from an adapted checkpoint. It reports every other tensor that changed in the adapted input. The strict 0% control again matches the foundation WAV exactly at SHA-256 `6330a6e2ed6d99a56c126522d0a87d91e0e12303257cd8bdfb63d3af1489c5cf`.

On `gtsja0165`, strict 5%, 10%, and 20% GYU mixtures preserve STT similarity `0.8372`, F0 p90 `34.18` to `34.69` cents, no clipping, and HF spikes below the matching source. However, their five-reference WavLM means remain `0.51078` to `0.51231` and ECAPA means decline from `0.08363` to `0.08122`. The safe GYU speaker row therefore has no meaningful identity benefit. The corrupted checkpoint is rejected, while the strict checkpoint is a diagnostic non-regression control only—not a GYU singer candidate.

Evidence:

- `artifacts/reports/diffsinger_gtsinger_heldout_set/diagnostic_directgyu_control_gtsja0165.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/diagnostic_directgyu_gtsja0165.json`
- `artifacts/reports/diffsinger_gtsinger_heldout_set/diagnostic_strict_gtsja0165.json`

This fix removes a false source of machine-like timbre but does not meet the identity goal. RC8, human acceptance, and OpenUtau packaging remain blocked on a non-destructive identity mechanism that improves both WavLM and ECAPA across the full five-phrase set.
