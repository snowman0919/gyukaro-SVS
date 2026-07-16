# RC9 OpenUtau song validation

Status: the first RC9 candidate failed human listening. A targeted replacement passes the objective gates and awaits explicit replacement listening. No final `v1.0.0` tag or release exists.

## Real OpenUtau path

Official upstream `https://github.com/stakira/OpenUtau.git` was freshly cloned at commit `27573ac5c888d927119d5f65a207312d79194b1f` on Ubuntu Linux aarch64 with .NET 8.0.422. Unmodified restore and build passed. Upstream tests were 205/206; the one JaPresamp failure reproduced alone before the GYU overlay and is unrelated. After applying the native GYU overlay, mapping tests passed 7/7 and the resident renderer test passed 1/1.

The local inferred score was validated and exported through the actual path:

```text
editable USTX
→ OpenUtau ValidateFull / RenderPhrase
→ native GYU renderer
→ resident gyu-singer-rc9
→ phrase WAV cache
→ OpenUtau RenderMixdown
→ 203.72 s stereo WAV
```

The replacement export contains 55 parts/phrases, 566 notes, and 566 generated phonemes with zero phoneme errors, failed requests, or retries. Cold render took 711.31 s; a repeat used all 55 cache entries in 0.256 s with no stale files.

## Isolated RC9 correction

The first full-song candidate exposed a real Japanese frontend defect: Katakana became inferred unknown phones and therefore incorrectly forced F0 to zero. NFKC/Katakana normalization fixed that root cause. A remaining phrase failure was then isolated across source, F0, SoulX settings, identity/style adapters, and refiners. The SoulX decoder tracked the score when the personalized residual was zero, but became unstable when the Korean-supervised residual was added to dense Japanese PITD. RC9 therefore keeps real-GYU personalized prosody Korean-only. English/Japanese use score plus user PITD and the generic SoulX singing prior. RC8 remains callable separately.

## Fixed-timeline objective result

No phrase stretching or optimized lag is used. Reference metrics use the local two-of-three F0 consensus; score timing/voicing metrics use the exact OpenUtau timeline.

| Metric | first candidate | human-failed RC9 candidate |
|---|---:|---:|
| reference F0 correlation | 0.9226 | 0.9254 |
| reference median cents | 20.88 | 12.62 |
| reference p90 cents | 125.95 | 113.44 |
| reference gross >600 cents | 5.15% | 4.65% |
| score F0 correlation | 0.9514 | 0.9550 |
| score p90 cents | 55.99 | 50.90 |
| score gross >600 cents | 1.97% | 1.60% |
| score voiced IoU | 0.8882 | 0.8836 |
| phrase onset median | 40 ms | 40 ms |

Those values describe the human-failed first RC9 candidate. The targeted replacement measures reference correlation 0.9290, reference p90 112.04 cents, reference gross >600 cents 4.64%, score correlation 0.9549, score p90 50.76 cents, and score gross >600 cents 1.64%. All objective gates pass. WavLM-to-GYU similarity remains non-regressed at mean 0.72939 and median 0.78667.

The replacement causally improves the long five-repeat phrase and the high refrain, but the ending refrain remains byte-identical and defective. The listening directory now includes before/after pairs for all three. Human listening remains mandatory; objective pass alone cannot promote the candidate.

Evidence: `artifacts/reports/openutau_upstream_v10.json`, `reference_song_rc9_runtime.json`, `reference_song_rc9_evaluation.json`, `reference_song_rc9_identity.json`, and `reference_song_rc9_listening_gate.json`.
