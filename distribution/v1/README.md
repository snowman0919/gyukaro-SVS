# GYU Singer v1.0 RC5

Production candidate for native multi-note GYU singing in OpenUtau. The runtime generates whole phrases with OmniVoice content, score/user-pitch conditioning, GYU prosody and latent identity/style adapters, and the SoulX-Singer decoder.

Linux with an NVIDIA CUDA GPU, Python 3.11+, Git, about 16 GB free RAM/unified memory, and about 20 GB free disk are required. The installer downloads roughly 10 GB of pinned model weights, including MMS forced alignment used for rapid and English phrase timing.

```sh
./install.sh
./launch-openutau.sh examples/openutau_v10_longform.ustx
```

The installer creates only `.runtime/` inside this folder, downloads exact revisions, builds the pinned OpenUtau fork automatically, and performs real KO/EN/JA 48 kHz render smokes. No manual clone, patch, virtualenv, `PYTHONPATH`, or model-cache configuration is needed.

See `INSTALL.md`, `OPENUTAU.md`, and `LIMITATIONS.md` before use.
