from __future__ import annotations

import argparse


def make_renderer(args):
    if args.backend == "loop":
        from .baseline_renderer import Renderer
        return Renderer(args.model)
    if args.backend == "neural-vocalizer-baseline":
        from .neural_renderer import NeuralRenderer
        return NeuralRenderer(args.model, args.audio_tokenizer, args.reference)
    if args.backend in {"hybrid-svs", "hybrid-soulx-phrase"}:
        from .inference import SoulXPhraseRenderer
        return SoulXPhraseRenderer(args.reference)
    from .inference import HybridRenderer, load_hybrid_model
    from .inference.codec import MossCodecDecoder
    return HybridRenderer(load_hybrid_model(args.checkpoint), MossCodecDecoder(args.audio_tokenizer), args.reference)


def main() -> None:
    parser = argparse.ArgumentParser(prog="gyu-singer")
    parser.add_argument("--backend", choices=("hybrid-svs", "hybrid-soulx-phrase", "hybrid-compact-experimental", "loop", "neural-vocalizer-baseline"), default="hybrid-svs")
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
    renderer = make_renderer(args)
    if args.command == "render":
        renderer.render_file(args.input, args.output); return
    from .renderer import build_server
    build_server(renderer, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
