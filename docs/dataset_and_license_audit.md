# Dataset and license audit

## License decision

Production experiments select only LibriTTS-R, VocalSet 1.2 (CC BY 4.0).
Excluded from distributed weights: Emilia original, JVS, GTSinger, JVS-MuSiC, SingNet. Emilia-YODAS is deferred pending item-level provenance review.
Raw external audio and the repaired local VocalSet archive are ignored and never bundled.

## Bounded quality-filtered subset

The reproducible selector considered 144 files and accepted 127 (0.296 hours): libritts_r 60/64, vocalset 67/80.
Accepted split counts are {'train': 71, 'validation': 28, 'test': 28}; speakers are disjoint across splits: True.
Measured gates cover clipping, DC offset, level, SNR proxy, high-frequency energy, duration, WavLM speaker consistency, and Whisper text agreement for LibriTTS-R.
VocalSet rows are isolated unaccompanied vocals; non-lexical technique clips have no fabricated ASR score. Music-background evidence is therefore source provenance, not a learned classifier.
Rejected-gate counts: {'asr_text': 3, 'clipping': 2, 'duration': 2, 'level': 1, 'snr_proxy': 6, 'speaker_consistency': 4}.

The official VocalSet archive checksum matches Zenodo, but its ZIP offsets overflow. A local `zip -FF` recovery copy is used only to decode selected WAVs, each validated with libsndfile.

Regenerate with `python scripts/build_external_dataset_registry.py`, `python scripts/prepare_external_acoustic_data.py`, then `python scripts/report_external_acoustic_data.py`.
