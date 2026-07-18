from __future__ import annotations

import argparse
import json
import sys

from .runtime_policy import RuntimePolicyError, load_backend_registry, resolve_backend


def make_renderer(args):
    if args.backend == "loop":
        from .baseline_renderer import Renderer
        return Renderer(args.model)
    if args.backend == "neural-vocalizer-baseline":
        from .neural_renderer import NeuralRenderer
        return NeuralRenderer(args.model, args.audio_tokenizer, args.reference)
    if args.backend in {"hybrid-svs", "hybrid-soulx-phrase", "orchestration-v0.4"}:
        from .inference import SoulXPhraseRenderer
        return SoulXPhraseRenderer(args.reference)
    if args.backend == "gyu-singer-v0.5":
        from .inference.v05 import GyuSingerV05Renderer
        return GyuSingerV05Renderer(args.reference)
    if args.backend == "gyu-singer-v0.6":
        from .inference.v06 import GyuSingerV06Renderer
        return GyuSingerV06Renderer(args.reference)
    if args.backend == "gyu-singer-v0.7":
        from .inference.v07 import GyuSingerV07Renderer
        return GyuSingerV07Renderer(args.reference)
    if args.backend == "gyu-singer-v0.8":
        from .inference.v08 import GyuSingerV08Renderer
        return GyuSingerV08Renderer(args.reference)
    if args.backend == "gyu-singer-rc5":
        from .inference.rc5 import GyuSingerRC5Renderer
        return GyuSingerRC5Renderer(args.reference)
    if args.backend == "gyu-singer-rc6":
        from .inference.rc6 import GyuSingerRC6Renderer
        return GyuSingerRC6Renderer(args.reference)
    if args.backend == "gyu-singer-rc8":
        from .inference.rc8 import GyuSingerRC8Renderer
        return GyuSingerRC8Renderer(args.reference)
    if args.backend == "gyu-singer-rc9":
        from .inference.rc9 import GyuSingerRC9Renderer
        return GyuSingerRC9Renderer(args.reference)
    if args.backend == "hybrid-compact-experimental":
        from .inference import HybridRenderer, load_hybrid_model
        from .inference.codec import MossCodecDecoder
        return HybridRenderer(load_hybrid_model(args.checkpoint), MossCodecDecoder(args.audio_tokenizer), args.reference)
    raise ValueError(f"backend has no renderer factory: {args.backend}")


def main() -> None:
    registry = load_backend_registry()
    parser = argparse.ArgumentParser(prog="gyu-singer")
    parser.add_argument("--backend", choices=tuple(registry["backends"]), default=None)
    parser.add_argument("--allow-experimental", action="store_true", help="explicitly run a non-production diagnostic backend")
    parser.add_argument("--model", default="checkpoints/gyu_v1_experimental.npz", help="loop or baseline vocalizer model")
    parser.add_argument("--checkpoint", default="checkpoints/gyu_hybrid_v0.2.pt")
    parser.add_argument("--audio-tokenizer", default="data/cache/moss-audio-tokenizer-nano")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("input")
    render.add_argument("--output", required=True)
    frontend = sub.add_parser("frontend")
    frontend.add_argument("--language", choices=("ko", "en", "ja"), required=True)
    frontend.add_argument("input", nargs="?", help="text to phonemize")
    frontend.add_argument("--text", help="text to phonemize (protocol-v2 documented form)")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "frontend":
        from .frontend import phonemize
        text = args.text or args.input
        if not text:
            parser.error("frontend requires text or --text")
        print(phonemize(args.language, text)); return
    try:
        decision = resolve_backend(args.backend, args.allow_experimental, registry)
    except RuntimePolicyError as error:
        parser.error(str(error))
    args.backend = decision.backend
    if decision.experimental_override:
        print("EXPERIMENTAL_OVERRIDE " + json.dumps({
            "backend": decision.backend,
            "status": decision.status,
            "reason": decision.reason,
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    renderer = make_renderer(args)
    if args.command == "render":
        renderer.render_file(args.input, args.output); return
    from .renderer import build_server
    server = build_server(renderer, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if hasattr(renderer, "close"): renderer.close()


if __name__ == "__main__":
    main()
