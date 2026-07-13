# Teacher evaluation status

Research verified current official interfaces and cloned pinned official source snapshots, but no teacher audio was admitted into v1. This prevents synthetic teacher identity from being misrepresented as GYU ground truth.

- Fish S2 Pro: 80+ languages, voice/reference conditioning, 44.1 kHz; official weights and code available. Runtime requires substantial VRAM. Source: https://huggingface.co/fishaudio/s2-pro and https://github.com/fishaudio/fish-speech.
- Higgs TTS 3 4B: multilingual, Transformers pipeline listed; current card is research/non-commercial licensed and is 9.32 GB. Source: https://huggingface.co/bosonai/higgs-tts-3-4b.
- MOSS-TTS Local Transformer v1.5: official code offers multilingual cloning, language tags, IPA control and finetuning; model family supports Korean, English, Japanese. Source: https://github.com/OpenMOSS/MOSS-TTS.

Pinned source inspections: Fish Speech `ad99ec5`, MOSS-TTS `e5e2926`, Higgs Audio `05a145b`. MOSS current v1.5 Local Transformer is 4B and 48 kHz stereo; it supersedes the previously named 1.7B non-v1.5 local checkpoint.

No pilot corpus, synthetic manifest, or quality scores exist because generation was not run. Required next gate: execute all three with a fixed 100×3 corpus, retain provenance/revisions, then reject samples on ASR, language ID, clipping, duration, F0, and two independent speaker embeddings.
