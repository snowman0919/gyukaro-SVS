# GYU Hybrid Singer v0.2 architecture

Stage 0 remains intact: `baseline_renderer.py` is source-loop rendering and `neural_renderer.py` is MOSS per-note TTS with pitch shift and phase vocoder. Both are baseline-only. They are not SVS claims and are never called by `hybrid-svs`.

`TriSingerModel` is compact phrase-level CFM. It receives the full frame sequence for all notes and phonemes in a score, then produces one codec-latent sequence. Frozen MOSS codec decoding converts that sequence to 48 kHz WAV. No waveform model is trained from scratch.

```text
protocol v2 score + lyric
  phonemize + phoneme-note frames + nominal/curve F0
  UnifiedPhonemeEncoder + LanguageFeatureEncoder + ScoreEncoder
  BlurredBoundaryEncoder + TimbreEncoder + StyleEncoder + PitchConditionEncoder
  ConditionalFlowTransformer (rectified-flow latent velocity)
  SingingDecoder latent bias
  frozen MOSS codec decoder
  phrase WAV
```

Real-anchor note timing is `inferred_from_speech_duration_not_ground_truth`. It is bootstrap control, never presented as real singing notation. Teacher speech is representation-only. Fish S2 probes are evaluation-only after license review. Accepted ACE-Step (Apache-2.0) then SoulX SVC WAVs are synthetic low-trust acoustic targets.

Model source: `src/gyu_singer/model/trisinger.py`. Phrase assembly: `alignment/phrase.py`. Inference: `inference/hybrid.py`. Losses: `losses/objectives.py`. Runtime: `renderer/service.py`.
