# RC6 objective quality evaluation

Status: HUMAN LISTENING FAIL; RC6 frozen; engineering resumed.

The actual `gyu-singer-rc6` backend rendered all nine stress cases. Against the fixed RC5 baseline, aggregate changes were: pitch MAE -0.002222 cents, voicing accuracy 0.001422, HF spike ratio -27.037611, spectral flux p95 -0.000572, sample jump p99.9 -0.002669, and ASR similarity 0. Clipping stayed zero.

The selected universal refiner lowers HF spikes and waveform jumps and slightly improves voicing. HF energy p95 rises by 0.000115; this is a remaining risk that objective metrics cannot adjudicate perceptually.

Resident stress passed with one unique hash across repeated renders, KO/EN/JA, concurrency, invalid-request recovery, restart, clean shutdown, and steady-state memory checks. The maintained OpenUtau overlay rendered 136 notes in 17 phrases over 119.982857 seconds with zero failed phrases.

The complete A-F production comparison selected D, the universal 25% refiner. E singing and F GYU adapters were rejected for pitch/voicing regressions. Actual pinned OpenUtau RC6 behavior passed note-pitch, lyric, PITD, style, cache-invalidation, KO/EN/JA, and phrase-render checks.

Listening gate: FAIL. The reviewer reports unnatural fade/staccato phoneme joins, buried syllables, rapid-case voice drift, metallic sound, inadequate score timing/pitch, and severe large-interval tearing. Final `v1.0.0` remains forbidden.
