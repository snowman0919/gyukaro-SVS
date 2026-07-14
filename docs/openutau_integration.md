# OpenUtau integration

Current official OpenUtau source is [`stakira/OpenUtau`](https://github.com/stakira/OpenUtau). Its documented extension points are editor plugins/phonemizers, voicebank renderers, and USTX project interchange; the official developer index also lists a proposed `svs.io` backend API. No stable generic external HTTP neural-renderer registration contract was found during the 2026-07-14 review.

OpenUtau USTX is UTF-8 YAML with voice-part note timing. `integrations/openutau/bridge.py` parses one voice part, exports renderer protocol v2 JSON and can POST it to resident `/render`, returning the generated WAV for cache/playback integration. It is unit-tested with tick-to-second conversion.

Executable smoke evidence: `examples/openutau_smoke.ustx` exported two Korean notes through `bridge.py`, posted to a live hybrid resident renderer, and returned a 48 kHz mono 49,920-frame WAV on 2026-07-14. The package includes this USTX fixture.

Native OpenUtau engine registration is therefore deliberately not claimed. Current blocker: bridge must either target a released `svs.io` contract or maintain a C# renderer extension against OpenUtau internals. Tempo maps and editor curves are not yet forwarded.
