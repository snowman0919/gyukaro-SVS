# RC8 local quality candidate

Status: **candidate 3 rejected; final diagnostic failure; not human-pending.**

The pass recorded below reflects a historical review that has been superseded. A later listening review reported excessive pitch and unintelligible output, and the final mandatory machine gate failed. RC8 candidate 3 is rejected and retained only as comparison evidence; RC7 is the accepted experimental baseline.

## Candidate 3 re-audit (2026-07-18)

The stronger stationary spectral correction was removed. Direct reconstruction of the frozen RC7 sustained file matched within `1.2e-7`, but temporal gain-mask smoothing did not reduce noise-floor modulation at short, medium, or long FFT resolution. A fixed-input SoulX sweep instead isolated CFG as the useful variable: on the neutral sustained case, `64 steps / CFG 1.5` retained the free-Whisper transcript and changed HNR by `+1.03 dB`, HF-energy p95 by `-29%`, and HF spike p99/median by `-34%` relative to `64 / 2.0`. The setting is limited to neutral Korean notes at least 2.5 seconds long; Rapid and Large Interval retain their previous policies.

The actual `gyu-singer-rc8` backend then rendered all nine cases into `artifacts/reports/rc8_candidate3_full/listening/`. Every file was analyzed directly with waveform/spectral metrics, RMVPE, free Whisper large-v3-turbo, WavLM, and ECAPA. The run is deterministic: an independent KO-neutral rerender has the same SHA-256 as the nine-file run.

| Metric | frozen RC7 | candidate 3 |
|---|---:|---:|
| Whisper lyric similarity | 0.924211 | 0.930667 |
| Pitch MAE, cents | 9.097778 | 7.686667 |
| Voicing accuracy | 0.868011 | 0.879822 |
| HF spike p99/median | 344.181322 | 362.179067 |
| Spectral flux p95 | 0.229099 | 0.230331 |
| Sample jump p99.9 | 0.073507 | 0.077264 |
| WavLM-to-GYU | 0.617891 | 0.612911 |
| ECAPA-to-GYU | 0.118516 | 0.114609 |

Candidate 3 is rejected, not `human_pending`. Sustained noise proxies improved, but aggregate HF spikes, discontinuity, and identity regressions fail the mandatory gate. EN and JA retain artifact trade-offs; Large Interval improves some F0/voicing/HF measures but loses speaker similarity. RC9 remains blocked.

### English decoder isolation

The EN sweep now preserves the production CTC content warp; its `s32_c1.5` WAV is byte-identical to candidate 3. Direct pre/post measurements show the SoulX output, not the refiners, is the main HF source (`2217.1755` before versus `1640.8127` after). `s32_c2` corrects free Whisper `them` to `the`, reduces the HF-spike proxy by 13.5%, reduces spectral flux by 7.1%, and slightly improves both speaker metrics. It also lowers HNR by 0.99 dB, raises flatness by 25%, raises sample discontinuity by 8.0%, and lowers voicing accuracy by 1.21 points. It is therefore a human A/B candidate, not a selected runtime change.

## Japanese duplicate-span probe (2026-07-18)

Status: **A rejected; diagnostic function is not connected to RC8.**

The held-out Japanese failure was reproduced before SoulX. Free Whisper on the full OmniVoice source returned `新しい歌を風に乗せて 新しい歌を風に乗せて届ける` (similarity `0.7222`). The bounded CTC probe required unknown ratio `<=0.10`, high-confidence target-phone coverage `>=0.85`, monotonic alignment, anchor log score `>=-2.0`, gap ratio `>=2.5`, and at least `0.50 s` unmatched excess. Held-out CTC was monotonic but had unknown ratio `0.2632` and coverage `0.6831`; quality JA had unknown ratio `0` but coverage `0.6036`. No source interval therefore qualified for removal. Both duplicate candidates fully fell back and are byte-identical to current RC8.

Identical F0, identity, style, decoder settings, and seed were used for the four approved paths. Each final file was directly analyzed with RMVPE, free Whisper, waveform metrics, short/medium/long STFT, WavLM, and ECAPA.

| held-out path | Whisper transcript | similarity | pitch MAE | voicing | HF spike | jump | WavLM | ECAPA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current RC8 | `新しい歌を風に乗せて新しい歌を風に乗せて届ける` | 0.7222 | 7.78 | 0.9189 | 1182.2560 | 0.075792 | 0.78279 | 0.21006 |
| global CTC 0.25 | same repeated phrase | 0.7222 | 7.27 | 0.9234 | 717.2484 | 0.083566 | 0.77626 | 0.25531 |
| chunked content, one SoulX decode | `新しい歌を風に乗せて届ける届ける` | 0.8966 | 7.88 | 0.8831 | 2195.0386 | 0.086425 | 0.82205 | 0.23699 |
| duplicate-span candidate | full fallback; same as current | 0.7222 | 7.78 | 0.9189 | 1182.2560 | 0.075792 | 0.78279 | 0.21006 |

The grouped source improves text coverage but misses the `0.90` gate and more than doubles the HF-spike proxy while reducing voicing. Global warp does not remove the repetition. The normal quality phrase remains protected: the candidate SHA equals current RC8, while global warp reduces lyric similarity from `0.8571` to `0.7857` and chunking raises HF spike from `186.6091` to `579.8170` with a large WavLM drop.

Evidence and WAVs: `artifacts/reports/rc8_ja_duplicate_span/evaluation.json`, `artifacts/reports/rc8_ja_duplicate_span/quality_ja/`, and `artifacts/reports/rc8_ja_duplicate_span/heldout_ja/`. The two `waveform_multires_stft.png` files contain waveform plus FFT-256/1024/4096 comparisons. All nine existing RC8 candidate WAV hashes still match their manifest. A and RC8 candidate 3 are rejected, and RC9 remains blocked.

### OmniVoice duration-collapse grid

After A failed, a fixed-seed 20-row grid varied five Japanese text lengths over `2.2/4.4/6.6/8.9 s`. Every source was checked with free Whisper, waveform metrics, and FFT-256/1024/4096 plots. At `<=0.5 s/character`, repetition occurred in `0/7` rows and mean lyric similarity was `0.9220`; at `>1.0 s/character`, repetition occurred in `5/7` and mean similarity fell to `0.7128`. The exact held-out text is correct at 4.4 and 6.6 seconds but repeats its prefix at 8.9 seconds (`0.7222`). `届ける` is correct at 2.2 seconds, repeats twice at 4.4, and repeats three times plus an unrelated token at 8.9.

Duration per character is therefore a strong risk factor, not a sufficient deterministic cause: one medium phrase fails at 6.6 seconds but not 8.9. A duration threshold cannot safely repair production. Evidence: `artifacts/reports/omnivoice_ja_duration_collapse/evaluation.json` and the corresponding listening/STFT files. This result permits investigation of a score-conditioned content replacement, but does not select or integrate one.

### DiffSinger JA content-source replacement probe

The cached GTSinger JA DiffSinger checkpoint was then tested as a content source only, with inferred phoneme timing forced into the existing score and one final SoulX phrase decode. This is not an RC8 runtime integration, and the checkpoint's CC BY-NC-SA 4.0 derivation is not release-compatible with the current package.

The held-out DiffSinger source removed the repeated prefix and reached free-Whisper similarity `0.9231`, but the identical-condition SoulX result fell to `0.8889`. The normal quality phrase regressed from current RC8 final similarity `0.8571` to `0.5714`; HF spike and sample-jump gates also failed. Pitch, voicing, identity, and all existing nine-file SHA checks passed, but those safety results do not offset the lexical and waveform failures. The candidate is rejected and is not callable from any renderer.

Evidence: `artifacts/reports/diffsinger_ja_content_source/evaluation.json`, including source/final transcripts, inferred phoneme timing, RMVPE, voicing, HF/sample-jump, WavLM/ECAPA, final WAV paths, and waveform plus FFT-256/1024/4096 plots.

### ACE-Step JA content-source replacement probe

The next cached replacement was ACE-Step v1 3.5B at revision `1bee4c9`, using fixed seed 101 and the requested phrase durations. It failed before SoulX: quality JA transcribed as `そう愛うたおう 小さな光を` (similarity `0.5185`) and held-out JA as `あたたしい歌を風に乗せて届ける` (`0.8571`). Neither meets the `0.90` source gate, so no final SoulX decode or runtime experiment was performed. Both source WAVs were still checked for clipping, discontinuity, HF behavior, and FFT-256/1024/4096 structure. Evidence: `artifacts/reports/ace_step_ja_content_source/source_evaluation.json` and `waveform_multires_stft.png`.

### Safe-duration OmniVoice latent-remap diagnostic

This diagnostic keeps OmniVoice and SoulX but generates the held-out lyric at the measured safe `6.6 s` duration, where free Whisper is exact, then monotonically maps its CTC-aligned content latent onto the `8.9 s` score timeline. The carrier is silence-padded only so SoulX produces the requested duration; the final output is still one phrase-level SoulX decode. There is no per-note TTS, final-WAV stitching, or waveform pitch shift. Diagnostic kana gives CTC unknown ratio `0`, high-confidence target-phone coverage `0.993258`, monotonic alignment, and mean log score `-0.981389`.

An initial same-worker fp16 check passed, so both paths were rerun in fp32 workers at 64 steps, CFG 2.0, seeds 7, 21, and 42 and then passed through the unchanged RC8 refiners. Candidate refined Whisper similarity was `0.96/1.00/1.00`, and repetition was absent at all three seeds. Pitch, voicing, HF spike, and sample jump also passed at every seed. Seed 21 initially appeared to improve identity, but that result did not generalize: candidate versus current WavLM/ECAPA was `0.76298/0.13816` versus `0.81089/0.19798` at seed 7 and `0.83212/0.21264` versus `0.84614/0.30710` at seed 42. Those are meaningful identity regressions, especially ECAPA.

Status is therefore `diagnostic_reject`, not human-pending. The route is not connected to RC8, quality JA and Rapid are untouched, and all nine frozen RC8 file hashes still match. The waveform and FFT-256/1024/4096 plots remain useful evidence that lexical continuity improved, but they do not override unstable GYU identity. Evidence and actual WAVs: `artifacts/reports/omnivoice_safe_duration_ja/evaluation.json`, `artifacts/reports/omnivoice_safe_duration_ja/waveform_multires_stft.png`, and `artifacts/reports/omnivoice_safe_duration_ja/multiseed/waveform_multires_stft.png`.

The failure is architectural rather than another CTC threshold issue. The frozen v0.7 identity ablation already measured only `+0.00200` mean WavLM and `+0.00134` mean ECAPA over no identity across six phrases, with held-out JA changing `-0.02389/-0.01070`. The bounded FiLM path therefore does not reliably override source-speaker information in SoulX. Duplicate removal, global warp, chunking, DiffSinger, ACE-Step, and safe-duration remapping are closed for this defect. Another content source should not be added until identity is conditioned inside a score-native singer or the SoulX identity path is retrained against multi-seed final-output speaker losses; either is an architecture-level change, not an RC8 local runtime patch.

### Truncated final-WAV identity gradient feasibility

Status: **K=2 and K=4 feasibility pass; zero optimizer steps; runtime unchanged.**

A fixed 3.0-second voiced crop was decoded with the production 64-step SoulX equations. Steps `64-K` ran under stop-gradient with the current adapter condition, the state was detached, and only the final K steps plus frozen vocoder stayed differentiable. The candidate started from the current v0.7 identity adapter. Identity-OFF repeated exactly, and both truncated variants were bit-identical to the current v0.7 WAV before backward.

| final differentiable steps | adapter grad norm | frozen parameter grads | peak allocated | peak reserved | backward-path runtime |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.000552965 | 0 | 12.481 GB | 12.753 GB | 10.4705 s |
| 4 | 0.001651947 | 0 | 20.654 GB | 21.070 GB | 11.1632 s |

All loss and audio tensors were finite. SoulX, vocoder, WavLM, and ECAPA parameters received no gradients; only `SoulXRealLatentAdapters.identity` did. Relative parameter drift was exactly zero because feasibility used no optimizer. The SoulX environment's older Transformers did not automatically migrate WavLM's legacy weight-normalization keys, so the probe maps the two shape-identical tensors and requires an empty missing/unexpected-key audit before use. Optional PEFT is disabled only inside this diagnostic because that venv's pinned Transformers predates the installed PEFT cache API.

The six emitted paths collapse to two unique WAV hashes: identity-OFF and current-v0.7/truncated. Free Whisper transcribed both as `아` with lyric similarity `1.0`; clipping was zero. Waveform and FFT-256/1024/4096 analysis is recorded in `artifacts/reports/truncated_identity_feasibility/audio_analysis.json` and `feasibility_unique_waveform_multires_stft.png`. K-specific evidence is in `k2/feasibility.json` and `k4/feasibility.json`.

This result authorizes only the next bounded optimizer diagnostic. It is not evidence of speaker improvement, does not create a human A/B candidate, and does not connect anything to RC8, RC9, OpenUtau, or packaging.

### Truncated final-WAV identity training and held-out gate

Status: **K=2 and K=4 diagnostic reject; experiment checkpoints deleted; runtime unchanged.**

The fixed input corpus contains three training phrases (`korean`, `english`, `japanese`), three validation phrases, five held-out phrases, and protected Rapid KO. Free Whisper passed every training/validation source with similarity at least `0.9286`. The known held-out Japanese OmniVoice failure remained excluded: its source repeated `新しい歌を風に乗せて`, scored `0.7222`, and was never loaded by training or identity promotion evaluation.

Both candidates started from the exact v0.7 adapter and ran 18 AdamW updates at `1e-4`, with SoulX, vocoder, WavLM, ECAPA, and the style adapter frozen. Every training decode used 64 total steps; only the final K steps were differentiable. Identity-OFF WAVs supplied waveform, FFT-256/1024/4096, frozen-content, and pitch-period preservation targets in addition to the two speaker losses and adapter/update regularizers.

| candidate | selected epoch | validation loss | max grad norm | max relative step | selected drift | peak allocated | training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=2 | 2 | 1.201825 | 0.040979 | 0.000806 | 0.004433 | 12.520 GB | 236.86 s |
| K=4 | 2 | 1.198866 | 0.084783 | 0.000816 | 0.004470 | 20.692 GB | 243.35 s |

No frozen parameter received a gradient, no tensor became non-finite, and non-identity adapter state stayed byte-identical. Training safety therefore passed, but safety was not treated as evidence of effectiveness.

The final matrix used the same source, target F0, reference, CFG, FP32 precision, and diffusion seed inside each comparison. Identity OFF, current v0.7, K=2, and K=4 were each rendered for all five held-out phrases plus protected Rapid KO at seeds 7/21/42: 72 actual one-pass phrase-level SoulX WAVs. Mean rendering time was 19.15 seconds per file. No per-note TTS, final-WAV stitching, or waveform pitch shifting was used.

| held-out mean | identity OFF | current v0.7 | K=2 | K=4 |
|---|---:|---:|---:|---:|
| WavLM-to-GYU | 0.609605 | 0.604288 | 0.605255 | 0.604923 |
| ECAPA-to-GYU | 0.107812 | 0.111448 | 0.111169 | 0.111036 |
| Whisper lyric similarity | 0.689982 | 0.718896 | 0.718896 | 0.718896 |
| RMVPE pitch MAE, cents | 46.5396 | 43.1837 | 43.7598 | 43.7438 |
| voicing accuracy | 0.834781 | 0.828301 | 0.827387 | 0.828732 |
| HF spike p99/median | 737.491 | 717.467 | 721.644 | 716.571 |
| sample jump p99.9 | 0.174625 | 0.170596 | 0.170691 | 0.170666 |

K=2 changed held-out WavLM/ECAPA by `-0.00435/+0.00336` versus identity OFF and `+0.00097/-0.00028` versus current v0.7. K=4 changed them by `-0.00468/+0.00322` versus OFF and `+0.00064/-0.00041` versus v0.7. Both therefore fail all four mandatory mean gates (`+0.01/+0.01` versus OFF and `+0.005/+0.005` versus v0.7). Phrase×seed pass ratios were only `6/18` for K=2 and `8/18` for K=4, and both failed protected Rapid KO.

Free Whisper also exposed failures that averages hide. Held-out KO remained exact across conditions, but held-out EN seed 7 became a long `Mmmm…` loop under every identity condition. At seed 21, identity OFF transcribed `Kyrie might play the boss, stay quiet driver`, while current/K=2/K=4 changed the ending to `traitor`. Rapid KO seed 7 changed from identity-OFF `아르기 너의 하자` to adapted `아르키 노예 하자`. Individual candidate failures additionally include pitch regression, voicing regression, HF-spike increases, and nonzero clipping samples. These are direct final-WAV results, not source-only proxies.

The rejected K=2 checkpoint SHA was `2da9aaebbfa2e5055a30aa3630d5db2110fa387dd2ad8711bce3664c38decf22`; K=4 was `acdfedfafeee6ca7c0659f512b66ae8fa35ba21fbd9e2f4484b97567f69caabc`. Both files were deleted after the gate failed. No human A/B candidate was created, and neither checkpoint is callable from the renderer or package.

Full sample-wise metrics, mean/median/minimum/standard deviation, transcripts, pass reasons, checkpoint disposition, and output paths are in `artifacts/reports/truncated_identity_evaluation/evaluation.json`. The 72 WAVs are under `artifacts/reports/truncated_identity_evaluation/listening/`; 18 matched-condition waveform plus FFT-256/1024/4096 comparisons are under `waveform_multires_stft/`. Training histories remain in `artifacts/reports/truncated_identity_training/k2/training.json` and `k4/training.json`. This closes the truncated-backprop diagnostic without changing RC8 and does not authorize RC9, OpenUtau, packaging, or release work.

## Scope and preserved baseline

RC7 remains frozen at `ae8944070f3dc38e310b33f29d95f4bcd3c81def`; its WAVs and checkpoint hashes are recorded in `docs/rc7_baseline.md`. RC8 writes only new artifacts and retains phrase-level SoulX decoding, 48 kHz PCM-24 output, the RC7 base spectral correction at strength 0.5, and the protected Rapid KO 64-step/CFG 2.0 policy. It uses no per-note TTS, waveform pitch shifting, or phase-vocoder note control.

## Diagnosis and bounded changes

| Defect | Isolated source | Selected RC8 change | Evidence |
|---|---|---|---|
| Sustained noise | Stable vowels benefited from stronger spectral correction; transient regions did not justify it | Extra 0.5 correction only inside Korean voiced runs that are stable for at least 300 ms, away from note/voicing boundaries, and only when a note lasts at least 2.5 s | `artifacts/reports/rc8_sustained_set/evaluation.json` |
| English transitions | ARPAbet `AY` was absent from the vowel set, forcing F0=0 through words such as *light* and *sky* | Classify `AY` as voiced and keep EN timing warp at 0.25 | `artifacts/reports/rc8_frontend_candidate/evaluation.json` |
| Korean overconnection | Generated-content/score phone-center mismatch was median 375 ms; warp 0.1 improved inferred closures but changed a lyric in the actual backend | Use only 0.05 warp, only on neutral contiguous non-interval KO; Rapid remains on its existing hold path | `artifacts/reports/rc8_ko_timing_sweep/evaluation.json` |
| Japanese attenuation | Note-by-note chunks missed phrase-only lexicon keys and became inferred unvoiced `ja_unknown_*` phones | Add the exact score chunks to the Japanese lexicon; do not amplify legitimate devoicing | `artifacts/reports/rc8_frontend_candidate/evaluation.json` |
| Large interval | The wrong apparent trajectory was already present before the spectral refiner: the 32-step decode tracked about 779 Hz in the first failure region | Use SoulX 50 steps/CFG 2.0 and do not apply the stronger stationary gate | `artifacts/reports/rc8_interval_candidate/evaluation.json` |

The first full RC8 attempt was rejected because over-broad timing/spectral gates reduced mean ASR to 0.732. The second attempt restored Rapid and Interval content but normal KO still changed *작은* to *큰*. The selected 0.05 KO warp preserves the full expected transcript. These rejected candidates remain under `artifacts/reports/rc8_backend_candidate/` and `artifacts/reports/rc8_backend_candidate2/` as negative evidence.

## Multi-resolution definitions

The repository had no prior SR/MR definition for this gate. RC8 defines three explicit 48 kHz analysis scales:

- short: FFT 256, hop 64, 5.33 ms window for attacks and consonants;
- medium: FFT 1024, hop 256, 21.33 ms window for phoneme/formant behavior;
- long: FFT 4096, hop 1024, 85.33 ms window for sustained harmonic stability.

On six sustained cases, strength 0.5 to the stationary-gated equivalent of strength 1.0 changed HNR proxy 22.63 to 23.41 dB, short/medium/long instability 0.02046/0.01316/0.03392 to 0.01866/0.01183/0.03095, and sample jump 0.03030 to 0.02231. Pitch MAE stayed 7.93 cents; voicing changed 0.9733 to 0.9707. This small voicing decrease was included in the subsequent mandatory listening review.

In the Large Interval failure region, RC7 to the selected 50-step candidate changed pitch MAE 15.42 to 11.41 cents, voiced accuracy 0.6625 to 0.6958, long-window instability 0.00856 to 0.00575, and long-window noise modulation 16.66 to 12.65 dB. The primary pYIN track changed from the erroneous 779 Hz trajectory to 387 Hz. A second estimator initially disagreed by roughly an octave, so the first candidate was rejected pending listening and a score-domain retest.

## Nine-file objective gate

RC7 and RC8 were evaluated against target F0 rebuilt with the current frontend.

| Metric | RC7 | RC8 candidate |
|---|---:|---:|
| ASR lyric similarity | 0.924211 | 0.930667 |
| ASR lyric coverage | 0.952189 | 0.952189 |
| Pitch MAE, cents | 9.097778 | 7.713333 |
| Voicing accuracy | 0.868011 | 0.873289 |
| Spectral flux p95 | 0.229099 | 0.226921 |
| Sample jump p99.9 | 0.073507 | 0.072940 |
| HF spike p99/median | 344.181322 | 362.049967 |
| WavLM-to-GYU | 0.617891 | 0.619059 |
| ECAPA-to-GYU | 0.118516 | 0.122516 |

This historical nine-file aggregate appeared to pass several isolated checks, but it did not achieve a global 5% HF-spike reduction and did not predict the later seed-specific lexical, clipping, Rapid, and identity failures. The final status is `rejected`, not `objective_nonregression_human_pending`.

Human review passed eight cases and rejected the first Large Interval candidate. A follow-up score-domain sweep found an 80 ms large-jump onset transition that keeps exact ASR while changing the failure-region pYIN/YIN disagreement from about 1203 to 4 cents. Relative to the failed RC8 interval file, pitch MAE improves from 11.41 to 10.55 cents, voicing from 0.6958 to 0.7292, and HF spike p99/median from 92.55 to 50.71. The actual-backend retest then passed listening and the bounded transition is integrated.

Validation: 49 tests passed; dataset validation passed for 132 recordings (`106..237`, mono 48 kHz, corrupt 0). The nine-file actual-backend render is the RC8 runtime smoke. Clean package validation belongs to RC9.
