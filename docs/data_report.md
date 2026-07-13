# GYU recording report

`scripts/index_recordings.py` measured 132 sequential ALAC `.m4a` files, indices 106–237. All decode to mono 48,000 Hz PCM S24LE masters. Total duration is 1,740.117 s (29.002 min), source duration range 3.669–41.216 s, median detected F0 136.075 Hz, median voiced-frame ratio 0.5031, and 2 takes have peak-clipping flags. Measured per-file duration, RMS, peak, approximate loudness, active ratio, autocorrelation F0 statistics, voiced ratio, clipping, and decode state live in `data/manifests/real_recordings.jsonl`.

Script PDF was found in source archive. Source order supports block assignment A–G: A 106–119, B 120–153, C 154–170, D 171–202, E 203–222, F 223–232, G 233–237. No ASR transcript was available, so every item has `script_text_unverified`, blank text, and confidence 0.35. This is intentionally not a usable text-aligned neural training set yet.

`real_segments.jsonl` has conservative active segments, real trust weight 1.0, and split labels. `script_alignment.jsonl` is a review artifact, not asserted correspondence.
