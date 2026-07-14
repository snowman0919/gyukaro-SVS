from __future__ import annotations

import io
import json
import traceback
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf


def build_server(renderer, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Build resident HTTP server. Renderer/model instance stays alive across requests."""
    # ponytail: one resident model worker; serialize until measured throughput needs a worker pool.
    render_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: dict, status: int = 200) -> None:
            body = json.dumps(value).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                health = renderer.health() if hasattr(renderer, "health") else {"status": "ok"}
                self._json(health, 200 if health.get("status") == "ok" else 503)
            elif self.path == "/model": self._json(renderer.model_info() if hasattr(renderer, "model_info") else {"backend": renderer.__class__.__name__, "sample_rate": renderer.sample_rate})
            else: self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/render": self.send_error(404); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                score = json.loads(self.rfile.read(size))
                output = io.BytesIO()
                with render_lock:
                    sf.write(output, renderer.render(score), renderer.sample_rate, format="WAV", subtype="PCM_24")
                body = output.getvalue()
                self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 400)
            except Exception as error:
                traceback.print_exc()
                self._json({"error": str(error), "type": type(error).__name__}, 500)

        def log_message(self, format: str, *args) -> None: pass
    return ThreadingHTTPServer((host, port), Handler)
