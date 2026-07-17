# OpenUtau DiffSinger quality reassessment

Overall status: **FAIL — no release**

The earlier v0.4/v0.6 outputs are not accepted as intelligible. The current work fixes two concrete score-native pipeline defects but does not yet produce a validated GYU voice.

## Proven root causes and fixes

1. The acoustic checkpoint trained only its deterministic auxiliary decoder, while the package requested diffusion depth 0.6. The stochastic path was untrained. The package now caps depth at 0.
2. Stock OpenUtau supplied nonzero score F0 through unvoiced consonants. A token-derived F0 mask is now embedded inside the acoustic ONNX, so no OpenUtau core fork is required.

On the same 20-note stock-OpenUtau phrase, Whisper similarity changed from `0.4571` to `0.9032` and pitch p90 error from `43.03` to `13.29` cents. Clipping remained `0.0`. The associated waveform/spectrogram/F0 review plot is generated beside the quality JSON.

## What failed

The foundation remains a GTSinger Japanese soprano voice. Four bounded identity attempts were rejected:

- learned speaker-embedding mixtures;
- low-register checkpoint interpolation;
- a 16,576-parameter phrase-paired neural mel residual;
- a gain-neutral 128-bin spectral-envelope direction.

None improved both WavLM and ECAPA against real GYU while preserving all other gates. Therefore none is enabled or described as GYU identity.

## Mandatory reporting rule

Every future audio candidate must include its SHA-256, free Whisper transcript/similarity, RMVPE pitch and voicing metrics, clipping, whole-file and sung-region waveform/spectral metrics, and a waveform/spectrogram/F0 plot. Human listening remains mandatory and cannot be replaced by these metrics.

No final tag or release is permitted from this state. Evaluation audio derived from the user-provided song is not packaged.
