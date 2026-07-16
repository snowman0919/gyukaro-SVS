Overall status: RC9 OBJECTIVE FAIL after mandatory free-STT and waveform audit
Highest achieved candidate: RC8 accepted; no RC9 listening or release candidate
Source commit: aa53ef4cc64391b845e28de713f2b4a8967a1c3a
RC7 baseline preserved: yes
RC8 quality gate: PASS and callable unchanged
Previous human listening: FAIL on repeated words, alignment, rapid/high phrases
Primary artifact source: SoulX content/F0 mismatch on long repeated lyrics with large jumps
Secondary artifact source: waveform refiners on high or large-jump Japanese phrases
OpenUtau full render: 31/31 phrases rendered, but intelligibility gate FAIL
Free Whisper: mean lyric similarity 0.6093 / p10 0.3333, FAIL
Waveform audit: no clipping; HF spike proxy 6023.73 and spectral flatness 0.08378
Score F0: correlation 0.9643 / p90 50.28 cents / gross >600 cents 1.59%
Reference F0: correlation 0.9281 / p90 119.41 cents / gross >600 cents 4.88%, PASS
GYU identity: non-regression PASS, WavLM mean 0.72839 / median 0.78667
Remaining audible risk: ending repetition count and tail retention require listening
Clean package: blocked; objective STT gate fails before human listening
Final v1.0.0 tag: prohibited

# RC9 human-failure engineering report

The earlier objective PASS did not include a free-STT gate and is invalid. The same
203.72-second OpenUtau render was re-evaluated phrase by phrase with Whisper plus
waveform/F0 analysis. Pitch and timing still pass, but lyric similarity averages
0.6093 and its tenth percentile is 0.3333, below the 0.75/0.50 release thresholds.
The renderer is therefore not intelligible enough to request another human release
review. Packaging now rejects any evaluation lacking per-phrase free transcripts and
waveform evidence.

The reported repetition is not a conventional autoregressive loop: OmniVoice uses iterative masked-token decoding and has no repetition-penalty control. Stage isolation showed that the long repeated source retained most lyrics, then collapsed when SoulX combined it with the large-jump target F0. Score-timed word chunks improved the production-path lyric score from 0.2667 to 0.75 while keeping pitch p90 essentially unchanged at 36.81 cents and improving voicing accuracy to 0.8997.

The high refrain exposed a separate causal defect. Latent identity/style ON versus OFF did not change lyric retention, while enabling the two waveform refiners reduced ASR similarity from 0.5333 to zero. RC9 now bypasses those refiners only for Japanese phrases at or above MIDI 80 or with a 12-semitone jump. RC8 remains unchanged.

Broad chunking was rejected: it damaged short refrains and a four-repeat phrase whose failed baseline transcript was already exact. Decoder steps/CFG, OmniVoice steps/guidance, unvoiced waveform mixing, and latent-adapter disable also failed to repair the ending refrain. A single 20 ms score-only onset relief improved kana-normalized diction similarity from 0.30 to 0.60 while keeping production pitch p90 at 125.34 cents. The ending before/after files are no longer identical, but complete repetition count and tail retention still require listening.

The continuity replacement joins only measured complete lyric-line groups and keeps the final high-speed repetitions as separate lines. It completed all 31 OpenUtau phrases without failures or retries. All pitch gates pass: reference gross >600 cents is 4.88%, score p90 is 50.28 cents, and score gross >600 cents is 1.59%. On identical complete-lyric evaluation groups, Whisper similarity improves from 0.4852 to 0.5821 and severe near-contiguous boundary troughs fall from 24 to 18. Those 18 troughs and the 1.82-second four-repeat line remain release blockers.

This does not erase the previous human FAIL. The replacement requires explicit listening of the new full song and the before/after pairs. Until that passes, RC9 is not achieved or packageable. No final tag or package was created.

Local listening files are under `data/external/work/rc9_reference/continuity_listening_candidate31/`. Reproducible causal metrics are indexed by `artifacts/reports/reference_song_rc9_continuity.json`, `reference_song_rc9_evaluation.json`, and `reference_song_rc9_human_failure_isolation.json`.
