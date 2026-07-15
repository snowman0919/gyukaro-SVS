# Singing prior training

The singing stage trained only the 4,968-parameter singing adapter for 800 steps on 18 VocalSet singing pairs plus 12 LibriTTS-R replay rows. Validation used 6 held-out rows and reached loss 2.795133.

VocalSet targets include rapid scales, arpeggios, vibrato/long tones, and straight reference excerpts. This stage was measured on production stress files, but it reduced high-frequency spikes while regressing voicing and ASR. It is therefore retained as a training baseline and disabled in the RC6 runtime.
