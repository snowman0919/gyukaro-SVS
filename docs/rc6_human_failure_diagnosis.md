# RC6 human-listening failure diagnosis

Status: **FAIL — RC6 is frozen and final v1.0.0 remains forbidden.**

The primary failure is the current phrase-SVC construction, not the final residual refiner. OmniVoice receives only the whole lyric and total duration. Rapid mode then maps the resulting phrase to the score by holding one CTC phone-center SoulX hidden vector across every target phoneme window at full strength. That operation explains the heard fade/staccato joins and rapid-case voice drift: it changes the content/timbre representation while removing natural within-phone evolution.

This is supported by the measured sweeps, not inferred from listening alone:

- Rapid: 18 decoder variants; status `no_strict_improvement`; ASR similarity [0.2857, 1.0]; HF-spike ratio [228.8973, 1989.4199].
- Large interval: 18 decoder variants; target voiced ratio [0.9875, 0.9875], observed [0.55, 0.6917]; pitch MAE [10.51, 126.77] cents.

Increasing SoulX steps, changing CFG/seed, generic denoising, or strengthening the post-filter is rejected. The next gate is a score-native phrase SVS probe in which phoneme duration, note onset, voicing, and F0 are direct acoustic-model inputs. It must first beat RC6 on rapid and large-interval timing, identity stability, and audible joins before any GYU adaptation or release-candidate packaging.
