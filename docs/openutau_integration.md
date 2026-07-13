# OpenUtau integration

OpenUtau USTX is UTF-8 YAML with 480 ticks per quarter note and voice-part note timing. `integrations/openutau/bridge.py` parses one voice part, exports renderer protocol v2 JSON and can POST it to resident `/render`. It was unit-tested with tick-to-second conversion. It is a bridge, not native OpenUtau engine registration; tempo maps and editor curves are not yet forwarded.
