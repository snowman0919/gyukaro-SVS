Overall status: FINAL DIAGNOSTIC FAILURE
RC7: accepted experimental baseline
RC8 candidate 3: rejected
Current v0.7 identity adapter: rejected for promotion
Truncated K=2: diagnostic_reject
Truncated K=4: diagnostic_reject
Production readiness: FAIL
RC9/OpenUtau/package/release: blocked
v1.0.0: prohibited

# Final diagnostic failure report

The mandatory machine gate failed. RC8 is not `human_pending`, and the generated listening WAVs are retained only as evidence of the failure modes. No checkpoint or experiment from RC8 candidate 3, current v0.7 identity adaptation, truncated K=2, or truncated K=4 is connected to a renderer, package, OpenUtau path, or release.

## Failure evidence

- English is seed-unstable: one seed collapses into repetition and other seeds produce different lyrics.
- Rapid KO mispronounces the lyric at seeds 7 and 42 and depends on seed 21.
- Large Interval target lyrics vary between forms such as `다가`, `나라`, and `나가`.
- Current v0.7 has RMVPE pitch MAE `43.18` cents, voicing accuracy `0.8283`, HF spike `717.47`, sample jump p99.9 `0.1706`, and nonzero clipping.
- Against identity OFF, current v0.7 changes WavLM by `-0.00532` and ECAPA by only `+0.00364`.
- Truncated K=2 and K=4 both fail the speaker-improvement, protected Rapid KO, and individual phrase/seed regression gates. K=2 passes only `6/18` held-out phrase×seed rows; K=4 passes `8/18`.

The authoritative sample-wise results, actual Whisper transcripts, RMVPE/voicing metrics, clipping, HF spikes, sample jumps, WavLM/ECAPA, waveform plots, and FFT-256/1024/4096 plots are preserved in:

- `artifacts/reports/truncated_identity_evaluation/evaluation.json`
- `artifacts/reports/truncated_identity_evaluation/listening/`
- `artifacts/reports/truncated_identity_evaluation/waveform_multires_stft/`
- `artifacts/reports/rc8_candidate3_full/`

Rejected experiment checkpoints were deleted after evaluation. Historical SoulX outputs remain diagnostic evidence only.

## Closed work on the current SoulX path

Do not continue identity-adapter training, increase truncated-backprop K, cherry-pick seeds, add strong spectral refinement, or use waveform pitch shifting on the OmniVoice-to-SoulX phrase path. These experiments did not establish stable lexical content, pitch/voicing, artifact safety, and identity at the same time.

## Score-native foundation gate

The next bounded step reproduced five evaluation-only `.ds` phrases from pinned GTSinger dataset-provided manual alignment and independently extracted RMVPE F0. All five source recordings pass free Whisper with similarity `1.0`. This is GTSinger evidence, not GYU supervision, and its CC BY-NC-SA 4.0 derivation is evaluation-only.

| candidate | all-phrase result | lyric mean / min | max pitch p90 | min voicing accuracy | max HF/source | WavLM / ECAPA |
|---|---:|---:|---:|---:|---:|---:|
| unadapted soprano | 5/5 qualified foundation only | 0.92444 / 0.8372 | 33.12 cents | 0.8808 | 1.070x | 0.54093 / 0.09710 |
| unadapted tenor | 0/5 reject | 0.62984 / 0.0000 | 28.38 cents | 0.8160 | 2.901x | 0.74056 / 0.21269 |
| existing GYU mix20 | 1/5 reject | 0.67422 / 0.4444 | 34.73 cents | 0.7943 | 3.023x | 0.73723 / 0.21152 |

The soprano depth-zero foundation produces different WAV hashes at seeds 7/21/42, but all `15/15` phrase×seed outputs pass free Whisper, pitch, voicing, clipping, and HF gates. It is therefore a qualified score-native lexical/pitch foundation only, not a GYU singer or RC. Tenor fails lexical, voicing, HF, or sample-jump gates on every phrase. The GYU mix fails four phrases and does not improve either WavLM or ECAPA over its tenor baseline (`-0.00333/-0.00117`).

Authoritative machine report: `artifacts/reports/diffsinger_gtsinger_heldout_set/aggregate_evaluation.json`. Per-phrase and per-seed reports are in the same directory and `seed_stability/`.

No new model training, identity adaptation, acoustic refiner, runtime change, package, tag, or release was performed. Production readiness remains `FAIL`.
