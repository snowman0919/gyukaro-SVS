# OpenUtau v0.5 readiness

Renderer protocol v2 now preserves notes, lyrics, tempo, pitch, dynamics,
breathiness, tension, brightness, vibrato, `prosody_strength`, and
`acoustic_style_strength`. Existing USTX bridge remains the integration surface.
Native OpenUtau phrase rendering is deferred to the next Goal; per-note bridge
fields not available from USTX remain absent rather than fabricated.
