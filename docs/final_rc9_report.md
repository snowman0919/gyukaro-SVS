Overall status: previous RC9 candidate HUMAN FAIL; replacement objective PASS; human listening pending
Highest achieved candidate: RC8 accepted; replacement RC9 remains a listening candidate
Source commit: bc93ff1624c1d7b2fb33545ecf1c4283dc093f91
RC7 baseline preserved: yes
RC8 quality gate: PASS and callable unchanged
Previous human listening: FAIL on repeated words, alignment, rapid/high phrases
Primary artifact source: SoulX content/F0 mismatch on long repeated lyrics with large jumps
Secondary artifact source: waveform refiners on high or large-jump Japanese phrases
OpenUtau full render: PASS technically, 55/55 phrases, zero failures and retries
Score F0: correlation 0.9549 / p90 50.76 cents / gross >600 cents 1.64%
Reference F0: correlation 0.9290 / p90 112.04 cents / gross >600 cents 4.64%, PASS
GYU identity: non-regression PASS, WavLM mean 0.72939 / median 0.78667
Remaining audible risk: short ending refrain unchanged
Clean package: blocked until replacement human listening PASS
Final v1.0.0 tag: prohibited

# RC9 human-failure engineering report

The reported repetition is not a conventional autoregressive loop: OmniVoice uses iterative masked-token decoding and has no repetition-penalty control. Stage isolation showed that the long repeated source retained most lyrics, then collapsed when SoulX combined it with the large-jump target F0. Score-timed word chunks improved the production-path lyric score from 0.2667 to 0.75 while keeping pitch p90 essentially unchanged at 36.81 cents and improving voicing accuracy to 0.8997.

The high refrain exposed a separate causal defect. Latent identity/style ON versus OFF did not change lyric retention, while enabling the two waveform refiners reduced ASR similarity from 0.5333 to zero. RC9 now bypasses those refiners only for Japanese phrases at or above MIDI 80 or with a 12-semitone jump. RC8 remains unchanged.

Broad chunking was rejected: it damaged short refrains and a four-repeat phrase whose failed baseline transcript was already exact. Decoder steps/CFG and OmniVoice steps/guidance sweeps also failed to repair the ending refrain. That region is byte-identical before and after and remains an explicit listening risk.

The replacement OpenUtau render completed all 55 phrases without failures or retries. After limiting the Japanese reading correction to the verified five-repeat repair, all objective gates pass: reference gross >600 cents is 4.64%, score p90 is 50.76 cents, score gross >600 cents is 1.64%, and WavLM identity mean is 0.72939.

This does not erase the previous human FAIL. The replacement requires explicit listening of the new full song and the before/after pairs. Until that passes, RC9 is not achieved or packageable. No final tag or package was created.

Local listening files and causal metrics are indexed by `artifacts/reports/reference_song_rc9_listening_gate.json` and `artifacts/reports/reference_song_rc9_human_failure_isolation.json`.
