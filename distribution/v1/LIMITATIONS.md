# Limitations

- Linux NVIDIA CUDA only was release-tested. Windows and macOS are unsupported in v1.0.
- The initial install downloads roughly 9 GB of model weights and needs substantial disk space.
- Peak measured memory on NVIDIA GB10 unified memory was about 15.4 GB above baseline for the complete resident/OpenUtau long-form run.
- Personalized singing prosody is Korean-only. EN/JA use generic multilingual prosody with GYU identity/style.
- Identity improvements are small and confidence intervals cross zero; outputs are not guaranteed to be indistinguishable from the target.
- Breathy and energetic have objective proxy validation. Soft, dark, and bright lack stable semantic calibration.
- Independent phrase rendering can contain intentional unvoiced consonant/coda windows up to 150 ms. The release test found no boundary clicks or harmful F0 jumps.
- Request cancellation reaches the HTTP/OpenUtau cancellation token, but the single serialized foundation-model worker cannot preempt a CUDA kernel already running.
- Automated metrics do not replace listening. The bundled listening manifest records whether human review was completed for this release candidate.
