from __future__ import annotations

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf

from .renderer import Renderer
from .neural_renderer import NeuralRenderer


def main() -> None:
    parser = argparse.ArgumentParser(prog="gyu-singer")
    parser.add_argument("--backend", choices=("loop", "neural"), default="neural")
    parser.add_argument("--model", default="data/cache/moss-tts-nano")
    parser.add_argument("--audio-tokenizer", default="data/cache/moss-audio-tokenizer-nano")
    parser.add_argument("--reference", default="data/source/Korea Digital Media High School 215.m4a")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("input")
    render.add_argument("--output", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    renderer = Renderer(args.model) if args.backend == "loop" else NeuralRenderer(args.model, args.audio_tokenizer, args.reference)
    if args.command == "render":
        renderer.render_file(args.input, args.output)
        os._exit(0)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/render":
                self.send_error(404); return
            score = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            output = io.BytesIO()
            sf.write(output, renderer.render(score), score.get("sample_rate", renderer.sample_rate), format="WAV", subtype="PCM_24")
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.end_headers()
            self.wfile.write(output.getvalue())
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
