# GTSinger held-out audio gate design

## Purpose

Replace the single rapid-phrase diagnostic with a small reproducible set that can expose pronunciation memorization before any DiffSinger candidate is promoted. This is evaluation infrastructure only; it does not change RC7, the production runtime, checkpoints, or packaging.

## Inputs

- Pinned local GTSinger metadata and source audio already under `data/external/`.
- Dataset-provided manual phoneme sequences and durations.
- Existing local RMVPE checkpoint and 20 ms DiffSinger F0 grid.
- Five held-out source rows whose source recordings achieved exact free-Whisper transcription during the upper-bound audit: `gtsja0165`, `gtsja0172`, `gtsja0174`, `gtsja0379`, and `gtsja0380`.
- Five real GYU identity references representing low, high, neutral, expressive, and long-phrase recordings: `212`, `215`, `216`, `219`, and `220`.

The selected GTSinger data is CC BY-NC-SA 4.0. Source audio remains external and is never committed or packaged.

## Data flow

One small builder reads the existing metadata, validates the matching manual alignment and source WAV duration, extracts RMVPE F0, and writes one `.ds` file per held-out phrase plus a manifest. Every row is labeled as dataset-provided alignment with RMVPE-derived target F0; it is not presented as GYU data.

The existing DiffSinger inference CLI renders the same five `.ds` files under each candidate condition. The existing evaluator then records SHA-256, free Whisper transcript, F0/voicing error, clipping, waveform and spectral metrics. WavLM and ECAPA similarities are measured against all five GYU references rather than selecting one favorable reference.

## Candidate gate

A candidate is rejected if any phrase lacks a valid source upper bound or if it materially fails any of these checks:

- free-Whisper lyric similarity at least `0.8` for every phrase;
- pitch p90 absolute error at most `100` cents and gross errors over `600` cents at most `5%`;
- no clipping and no high-frequency spike regression beyond the existing reference-relative bound;
- no material loss against the unadapted foundation in voiced timing or waveform discontinuity;
- identity improvement must be consistent across the five GYU references, not a single-reference outlier.

Passing these automated checks creates only a human-listening candidate. It does not authorize RC8, RC9, packaging, or release.

## Failure handling

Missing source audio, mismatched phoneme duration, an invalid F0 grid, or a source Whisper upper bound below the gate stops that row instead of fabricating or time-warping evidence. Failed candidates remain diagnostic artifacts and are not wired into OpenUtau.

## Verification

The builder must regenerate identical row IDs and durations from the pinned inputs, leave `data/source/` unchanged, and pass its duration/F0 assertions. The candidate evaluation must run free Whisper, RMVPE, and waveform analysis on every generated WAV. Repository dataset validation remains required before any later runtime promotion.
