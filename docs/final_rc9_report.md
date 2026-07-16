Overall status: RC9 objective pass; human listening pending
Highest achieved candidate: RC8 accepted; RC9 objective candidate
Source commit: 284daec4096d5ffa1e95682284a6cce53f180e2f
RC7 baseline preserved: yes, commit and artifacts unchanged
RC8 quality gate: PASS, including final human listening
Rapid KO: retained unchanged
Sustained KO: RC8 PASS retained
English: RC8 PASS retained; RC9 uses score/user PITD plus generic prosody
Japanese: objective full-song candidate ready; human listening pending
Large Interval KO: RC8 accepted backend retained; excluded from RC9 song-policy change
OpenUtau upstream: official clean clone built; unrelated upstream test 205/206
OpenUtau commit: 27573ac5c888d927119d5f65a207312d79194b1f
Native GYU renderer: PASS, actual RenderMixdown path
Reference song local analysis: PASS, inferred labels explicitly marked
OpenUtau reference project: PASS locally; excluded from distribution
Full OpenUtau song render: PASS technically, 55/55 phrases, human listening pending
F0 agreement: PASS, reference corr 0.9254 / median 12.62c / p90 113.44c / gross >600c 4.65%
Timing agreement: PASS, score p90 50.90c / voiced IoU 0.8836 / onset median 40ms
GYU identity: objective PASS, WavLM median 0.79529; human confirmation pending
Clean RC9 package: blocked by mandatory human listening gate
Package SHA-256: pending

# Final RC9 candidate report

The RC9 engineering candidate satisfies the real OpenUtau and objective song gates. The only remaining authorized next action is human listening of the local full-song gate. Packaging and final achievement remain blocked until that explicit review passes. Final `v1.0.0` remains prohibited.
