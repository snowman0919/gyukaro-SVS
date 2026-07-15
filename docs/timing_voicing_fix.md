# Timing and voicing fix

RC4's primary artifact source was the interaction of score/content timing mismatch and nonzero F0 in silence or unvoiced consonants. RC5 introduced one 50 Hz phrase timeline carrying phoneme and note bounds, voicing class, nominal score F0, validated GYU residual, and editor pitch. Silence and unvoiced consonants receive F0=0. Vowels and voiced consonants receive score F0 plus prosody and user PITD.

The content path remains phrase-level. Rapid Korean uses MMS CTC phoneme-hold mapping inside SoulX content hidden state; English uses the measured 0.25 latent timing correction. Unedited slurs receive only a minimal 60 ms transition; explicit OpenUtau pitch remains authoritative and hard attacks are preserved. No per-note TTS, phase vocoder, or waveform pitch shifting is used.

The isolation matrix and rejected timing variants are recorded in `docs/rc4_artifact_isolation.md` and `artifacts/reports/rc5_isolation/`.
