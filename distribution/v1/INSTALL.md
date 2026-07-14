# Installation

Supported release target: Linux `x86_64` or `aarch64`, NVIDIA CUDA GPU, Python 3.11+, Git, and network access. If `dotnet` 8 is absent, the installer downloads a private SDK under `.runtime/dotnet`.

```sh
unzip gyu-singer-v1.0.zip
cd gyu-singer-v1.0
./install.sh
```

The default flow creates isolated environments, downloads and verifies pinned OmniVoice and SoulX weights, clones the exact OpenUtau source revision, applies the GYU overlay, builds it, and renders KO/EN/JA smoke WAVs under `.runtime/`.

An existing exact development cache can seed an offline/local validation without copying virtual environments:

```sh
./install.sh --cache-source /absolute/path/to/data/cache
```

Rerunning the installer is safe: verified model files and the completed OpenUtau overlay are reused.

Start the editor:

```sh
./launch-openutau.sh examples/openutau_v09.ustx
```

For a headless render or resident service:

```sh
./render-example.sh examples/quality_ko.json output.wav
./serve.sh
```
