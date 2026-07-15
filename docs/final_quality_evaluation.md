# RC6 objective quality evaluation

Status: HUMAN LISTENING FAIL; RC6 frozen; engineering resumed.

The actual `gyu-singer-rc6` backend rendered all nine stress cases. Against the fixed RC5 baseline, aggregate changes were: pitch MAE -0.002222 cents, voicing accuracy 0.001422, HF spike ratio -27.037611, spectral flux p95 -0.000572, sample jump p99.9 -0.002669, and ASR similarity 0. Clipping stayed zero.

The selected universal refiner lowers HF spikes and waveform jumps and slightly improves voicing. HF energy p95 rises by 0.000115; this is a remaining risk that objective metrics cannot adjudicate perceptually.

Resident stress passed with one unique hash across repeated renders, KO/EN/JA, concurrency, invalid-request recovery, restart, clean shutdown, and steady-state memory checks. The maintained OpenUtau overlay rendered 136 notes in 17 phrases over 119.982857 seconds with zero failed phrases.

The complete A-F production comparison selected D, the universal 25% refiner. E singing and F GYU adapters were rejected for pitch/voicing regressions. Actual pinned OpenUtau RC6 behavior passed note-pitch, lyric, PITD, style, cache-invalidation, KO/EN/JA, and phrase-render checks.

Listening gate: FAIL. The reviewer reports unnatural fade/staccato phoneme joins, buried syllables, rapid-case voice drift, metallic sound, inadequate score timing/pitch, and severe large-interval tearing. Final `v1.0.0` remains forbidden.

## RC7 spectral-refiner candidate

Status: objective gate passed; human listening pending; no runtime integration, tag, or release.

The aligned spectral-singing refiner at 50% is the first post-RC6 probe to preserve nine-file ASR while materially reducing aggregate artifacts. Compared with RC6: HF spikes -24.9%, sample jumps -21.5%, spectral flux -1.2%, pitch MAE -0.23 cents, voicing +0.0007, WavLM-to-GYU +0.0179, and ECAPA-to-GYU +0.0065. Rapid lyrics remain `빠르게 노래하자`; large-interval lyrics remain `높이 날아`.

The large-interval case remains the risk: sample jumps and pitch improve, but its HF-spike ratio rises 10.8%. Therefore automated metrics do not declare the artifact fixed. The compact nine-file and RC6-before/after gate is `artifacts/reports/rc7_listening_gate/`. Final `v1.0.0` remains forbidden until explicit listening passes and the accepted model is integrated and packaged.
