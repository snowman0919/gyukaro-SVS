# Dataset and license audit

## License decision

Production experiments select only LibriTTS-R, VocalSet 1.2, Zeroth-Korean SLR40 (CC BY 4.0).
Excluded from distributed weights: Emilia original, JVS, GTSinger, Children's Song Dataset, JVS-MuSiC, SingNet. Emilia-YODAS is deferred pending item-level provenance review.
Raw external audio and the repaired local VocalSet archive are ignored and never bundled.

## Bounded quality-filtered subset

The reproducible selector considered 144 files and accepted 127 (0.296 hours): libritts_r 60/64, vocalset 67/80.
Accepted split counts are {'train': 71, 'validation': 28, 'test': 28}; speakers are disjoint across splits: True.
Measured gates cover clipping, DC offset, level, SNR proxy, high-frequency energy, duration, WavLM speaker consistency, and Whisper text agreement for LibriTTS-R.
VocalSet rows are isolated unaccompanied vocals; non-lexical technique clips have no fabricated ASR score. Music-background evidence is therefore source provenance, not a learned classifier.
Rejected-gate counts: {'asr_text': 3, 'clipping': 2, 'duration': 2, 'level': 1, 'snr_proxy': 6, 'speaker_consistency': 4}.

The official VocalSet archive checksum matches Zenodo, but its ZIP offsets overflow. A local `zip -FF` recovery copy is used only to decode selected WAVs, each validated with libsndfile.

Zeroth-Korean contributes bounded four-speaker Korean speech controls: 400 utterances (1.025 hours) and a non-duplicated 1,400-utterance scale probe (3.544 hours). MMS-CTC timings are explicitly inferred; these rows are neither singing nor GYU ground truth. The scale probe failed lexical transfer and is retained only as negative evidence.

Two external pretrained SVC candidates were audited but excluded before large downloads. HQ-SVC's repository/checkpoint label is Apache-2.0, while its paper identifies OpenSinger and M4Singer training data; the unresolved non-commercial training-data provenance fails the production checkpoint gate. CopyCat's official model card is PolyForm Noncommercial 1.0.0. Neither model nor derived output enters production training.

The CSD paper footer states CC BY 4.0, but the actual Zenodo record `4916302` reports `cc-by-nc-sa-4.0` for the downloadable 1.85 GB archive. The archive metadata is used as the conservative controlling evidence. CSD data, replay training, and CSD-derived production checkpoints are excluded; an MIT code or model-repository label does not erase this data-license conflict.

Regenerate with `python scripts/build_external_dataset_registry.py`, `python scripts/prepare_external_acoustic_data.py`, then `python scripts/report_external_acoustic_data.py`.
