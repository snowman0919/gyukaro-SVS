# RC9 continuity recovery

Status: objective improvement; human listening pending; release blocked.

The uploaded mixed file was an aligned instrumental plus the existing OpenUtau vocal. Its vocal residual correlated 0.999876 with the prior local render, so mixing was not the source of the chopping. The score builder had created 55 independent lyric-line parts, and OpenUtau therefore made 55 independent neural requests. At near-contiguous lyric boundaries, 24 of 40 had an energy trough below 15% of their surrounding audio.

Blindly joining every line was rejected. An 89-second request reduced boundary troughs but collapsed content, lowering the grouped Whisper lyric score from 0.4567 to 0.2740. The accepted engineering candidate uses fixed-seed isolation over complete lyric lines, retains short semantic requests, and uses score/phoneme/PITD context plus a 100 ms overlap only inside requests longer than 12 seconds. It remains a local reference-song test; the inferred score, lyrics, project, stems, and audio are excluded from Git and packages.

| Measure | Previous candidate | Continuity candidate |
|---|---:|---:|
| OpenUtau phrases | 55 | 31 |
| Whisper weighted lyric similarity | 0.4852 | 0.5821 |
| Severe near-contiguous boundary troughs | 24 / 40 | 18 / 40 |
| Median boundary energy ratio | 0.0447 | 0.2276 |
| Score F0 correlation | 0.9548 | 0.9643 |
| Score F0 p90 | 52.03 cents | 50.28 cents |
| Score gross error over 600 cents | 1.64% | 1.59% |

This is not a release pass. The four `息が詰まる` repetitions compressed into 1.82 seconds still lose diction, and 18 severe boundary troughs remain. Generating each repeat independently and generating a longer source followed by pitch-preserving time compression both regressed ASR; those experiments are rejected and absent from the runtime.

MoonInTheRiver DiffSinger was inspected as the requested architectural reference, and OpenVPI DiffSinger defines the relevant production contract: an acoustic SVS accepts phoneme sequence, phoneme duration, and F0 directly. The repository's existing direct GYU DiffSinger pilots satisfy that interface but fail Korean lexical quality (ASR 0.148–0.311 on bounded stress probes). They are not packaged as a fake OpenUtau DiffSinger voicebank. A native DiffSinger/OpenUtau release requires a compatible, redistributable lexical singing foundation followed by GYU adaptation that passes rapid diction, identity, and human listening gates.

Local listening files:

- `data/external/work/rc9_reference/continuity_listening_candidate31/01_vocal_only.wav`
- `data/external/work/rc9_reference/continuity_listening_candidate31/02_full_mix.wav`
- `data/external/work/rc9_reference/continuity_listening_candidate31/02_full_mix.mp3`

No RC package, GitHub release, or final tag is authorized by this milestone.
