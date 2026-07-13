# Teacher evaluation status

Research verified current official interfaces and cloned pinned official source snapshots. No teacher audio was admitted into v1. This prevents synthetic teacher identity from being misrepresented as GYU ground truth.

- Fish S2 Pro: 80+ languages, voice/reference conditioning, 44.1 kHz; official weights and code available. Runtime requires substantial VRAM. Source: https://huggingface.co/fishaudio/s2-pro and https://github.com/fishaudio/fish-speech.
- Higgs TTS 3 4B: multilingual, Transformers pipeline listed; current card is research/non-commercial licensed and is 9.32 GB. Source: https://huggingface.co/bosonai/higgs-tts-3-4b.
- MOSS-TTS Local Transformer v1.5: official code offers multilingual cloning, language tags, IPA control and finetuning; model family supports Korean, English, Japanese. Source: https://github.com/OpenMOSS/MOSS-TTS.

Pinned source inspections: Fish Speech `ad99ec5`, MOSS-TTS `e5e2926`, Higgs Audio `05a145b`. MOSS current v1.5 Local Transformer is 4B and 48 kHz stereo; it supersedes the previously named 1.7B non-v1.5 local checkpoint.

Executed requested-teacher result: all three requested teachers ran with the same authorized GYU reference (`gyu_real_000215`), its corresponding Korean reference transcript, and the Korean prompt `하늘에 빛이 내려와.`.

| Teacher | Output | Measured artifact | Status |
|---|---|---|---|
| Fish S2 Pro | `data/teacher/fish_s2_ko.wav` | 44.1 kHz mono, 2.786395 s, peak 0.206116, RMS 0.034635, silence 0.378695 | Unadmitted |
| Higgs TTS 3 4B | `data/teacher/higgs_tts3_ko.wav` | 24 kHz mono, 4.52 s, peak 0.914032, RMS 0.193381, silence 0.242754 | Unadmitted |
| MOSS Local Transformer v1.5 | `data/teacher/moss_local_ko.wav` | 48 kHz stereo, 3.92 s, peak 0.241180, RMS 0.046807, silence 0.204953 | Unadmitted |

`scripts/evaluate.py` ran local Whisper large-v3-turbo, WavLM base-plus speaker verification, ECAPA-TDNN speaker verification, acoustic/F0 checks, and pairwise WavLM teacher agreement on these three controlled outputs. All transcribed `하늘에 빛이 내려와` exactly after normalization (content/language score 1.0). WavLM/ECAPA/reference scores were Fish `0.9167/0.6910`, Higgs `0.8899/0.7139`, MOSS Local `0.8122/0.5865`; teacher-agreement scores were `0.8977`, `0.8600`, and `0.8800`. The machine-readable result is `artifacts/eval/teacher_scored.jsonl`.

These are **teacher-gate passes, not training admission**: three Korean short prompts cannot establish 100×3 language coverage, sustained-singing quality, or resistance to teacher identity leakage.

Executed pilot: Apache-2.0 `OpenMOSS-Team/MOSS-TTS-Nano` replacement (0.1B) ran in cloned GYU voice mode for one Korean, English, and Japanese phrase. `teacher_pilot.jsonl` and `teacher_filtered.jsonl` retain provenance. All three pass only basic acoustic gates: 6.4 s, 48 kHz stereo, peak 0.447–0.845, RMS 0.082–0.182, silence 0.020–0.055. They remain **unadmitted** because speaker-embedding, ASR/CER, language-ID, and cross-teacher disagreement gates are not yet run.

Higgs TTS 3 4B and MOSS Local Transformer v1.5 were downloaded and successfully served through SGLang-Omni in an isolated Python 3.12 environment. On GB10, the MOSS codec needed its SDPA fallback because SGLang FA3 cannot fall back to a FA2 package in this environment; the fallback is logged and used only for the teacher pilot. MOSS-Nano remains a clearly labeled small replacement pilot, not evidence of Local-Transformer fine-tuning.

The fixed 100×3 corpus is `configs/teachers/trilingual_pilot.jsonl`; it rotates five categorized real GYU references and five style labels per language.

## Full MOSS Local v1.5 run

The complete 300-item corpus was generated with `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5@be7766a6735b98bd793f7c79fb720b4d0f5d13b8`. The provenance manifest is `data/manifests/teacher_moss_local_v15.jsonl`; generated WAVs remain ignored synthetic artifacts. `artifacts/eval/teacher_moss_local_v15_scored.jsonl` records the real gate output: 196/300 passed the acoustic, Whisper content/language, WavLM, and ECAPA thresholds, while 104/300 require review.

| Language | Generated | Gate pass, unadmitted | Mean confidence |
|---|---:|---:|---:|
| Korean | 100 | 99 | 0.9050 |
| English | 100 | 78 | 0.8204 |
| Japanese | 100 | 19 | 0.7483 |

## Full cross-teacher gate and weighted corpus

Fish S2 Pro and Higgs TTS 3 4B completed the same pinned 100×3 corpus. `teacher_full_trilingual.jsonl` combines all 900 outputs; `teacher_full_trilingual_scored.jsonl` applies Whisper, WavLM, ECAPA, acoustic checks, and same-item cross-teacher agreement. 633/900 pass without training admission; 267 require review.

| Language | Gate pass | Weighted representation rows | Mean trust weight |
|---|---:|---:|---:|
| Korean | 291 | 291 | 0.1901 |
| English | 273 | 273 | 0.1841 |
| Japanese | 69 | 69 | 0.1777 |

The selected manifest is `data/manifests/teacher_weighted.jsonl`: MOSS 196, Fish 228, Higgs 209. Every row is `representation_distillation_only_not_singing_decoder`; no synthetic teacher speech is presented as real GYU singing. The fixed core corpus has neutral, soft, breathy, energetic, and bright conditions; `trilingual_style_supplement.jsonl` supplies separately tracked dark/emotional cases for the next teacher pass.
