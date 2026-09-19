"""Zero-dependency local server for the visualizer.

Uses http.server so `python -m nanovllm.viz` works without installing anything
beyond what the engine already needs.
"""

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nanovllm.viz import models
from nanovllm.viz.engine import run_trace


STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # the console belongs to the trace output, not request noise

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/models":
            return self._json(200, models.to_json())
        name = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / name).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return self._json(404, {"error": f"not found: {name}"})
        self._send(200, target.read_bytes(), TYPES.get(target.suffix, "application/octet-stream"))

    def do_POST(self):
        if self.path.split("?")[0] != "/api/run":
            return self._json(404, {"error": "unknown endpoint"})
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            return self._json(400, {"error": f"bad JSON: {exc}"})

        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            return self._json(400, {"error": "Enter a prompt to trace."})
        try:
            trace = run_trace(
                req.get("model", models.MOCK_ID),
                prompt,
                max_tokens=int(req.get("maxTokens", 24)),
                temperature=float(req.get("temperature", 0.6)),
                block_size=int(req.get("blockSize", 8)),
                num_blocks=int(req.get("numBlocks", 48)),
                max_num_batched_tokens=int(req.get("maxNumBatchedTokens", 32)),
                apply_chat_template=bool(req.get("chatTemplate", True)),
            )
        except Exception as exc:
            traceback.print_exc()
            return self._json(400, {"error": str(exc) or exc.__class__.__name__})
        print(f"traced {len(trace['frames'])} steps - {trace['model']['label']}")
        return self._json(200, trace)


def serve(host: str = "127.0.0.1", port: int = 8008, open_browser: bool = True):
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"nano-vllm visualizer: {url}")
    problem = models._missing_runtime()
    print("  real models:", problem if problem else "ready")
    print("  mock engine: ready (real scheduler, fake forward pass)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()
