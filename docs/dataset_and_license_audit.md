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

FM-Singer was audited as a Korean score-native diagnostic. Its code repository is MIT, but the official generator checkpoint is derived from the AI Hub multi-speaker singing corpus and no checkpoint/data redistribution permission was established. The 2.09 GB checkpoint was therefore used only for local evaluation and is not a production or package dependency. The only usable checkpoint speaker, AMS14, failed the large-interval lyric gate; the other bounded AMS speaker embeddings were mostly silent or unusable.

MeloTTS Korean was audited as a model dependency, not a dataset. The official repository and Korean model card both declare MIT, and the local checkpoint SHA-256 is `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0`. It was used only for a bounded interface diagnostic. Exact score-duration forcing reduced mean stress ASR from 0.467 to 0.133, and the model has no explicit score-F0 control. It is excluded from production weights and packages.

Seed-VC was audited as an evaluation-only SVC dependency. The official repository revision `51383ef` and model revision `257283f` declare GPL-3.0, while the downloaded 44.1 kHz F0 checkpoint has SHA-256 `42aef93ffe65857c840d270252fa040f7ba04514945ec460f3ac1ac2a96de684`. Its training-data provenance is not documented, and the tested score-source checkpoint is CSD-derived. Seed-VC and every derived conversion are excluded from production training, packages, and runtime integration.

The CSD paper footer states CC BY 4.0, but the actual Zenodo record `4916302` reports `cc-by-nc-sa-4.0` for the downloadable 1.85 GB archive. The archive metadata is used as the conservative controlling evidence. CSD data, replay training, and CSD-derived production checkpoints are excluded; an MIT code or model-repository label does not erase this data-license conflict.

VocalSet scale-up reused the already verified CC BY 4.0 archive. A quality gate selected 221 clips (0.455 h), spanning 20 speaker-disjoint train/validation/test singers and 12 singing techniques. LibriTTS-R scale-up used all 52 non-reference accepted clips across eight speaker-disjoint speakers. Every v2 degradation input is an actual FP32 SoulX reconstruction; random-noise corruption remains unused.

Regenerate with `python scripts/build_external_dataset_registry.py`, `python scripts/prepare_external_acoustic_data.py`, then `python scripts/report_external_acoustic_data.py`.
