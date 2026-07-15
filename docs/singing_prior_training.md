# Singing prior training

The singing stage trained only the 4,968-parameter singing adapter for 800 steps on 18 VocalSet singing pairs plus 12 LibriTTS-R replay rows. Validation used 6 held-out rows and reached loss 2.795133.

VocalSet targets include rapid scales, arpeggios, vibrato/long tones, and straight reference excerpts. This stage was measured on production stress files, but it reduced high-frequency spikes while regressing voicing and ASR. It is therefore retained as a training baseline and disabled in the RC6 runtime.

The expanded v2 experiment increased real SoulX-to-clean VocalSet pairs from 24 total/18 train to 221 total/130 train, covering 20 singers and 12 technique groups with speaker-disjoint validation and test splits. A 25% adapter strength clearly improved held-out spectral and high-band reconstruction, but the RC stress path still traded a small HF reduction for worse voicing. At 15%, ASR matched RC6 while the HF-spike improvement was only 4.6%. The v2 singing adapter is therefore also disabled by default.
