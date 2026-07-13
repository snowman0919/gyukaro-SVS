# Hybrid architecture v0.2

`TriSingerModel` processes an entire phrase frame sequence once. It combines unified phonemes, language features, score, blurred boundaries, reference timbre, style controls and explicit F0 conditions. `ConditionalFlowTransformer` predicts a rectified-flow acoustic latent velocity. `SingingDecoder` adds a learned latent bias; frozen MOSS audio-tokenizer decoder converts 768-D latents to 48 kHz audio. No waveform decoder is trained from random initialization.

Real-anchor score timing is inferred from speech duration and labelled `inferred_from_speech_duration_not_ground_truth`; it is a bootstrap control signal, not real singing annotation. Stage 0 `baseline_renderer.py` and `neural_renderer.py` are retained only as baselines. Hybrid inference never calls their note loop, pitch shift or phase vocoder.
