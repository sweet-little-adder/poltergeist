"""Small Vercel Python function using the same deterministic research engine as the CLI."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from poltergeist.agent import answer  # noqa: E402
from poltergeist.store import RunStore  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
            self.send_error(413)
            return
        try:
            payload = json.loads(self.rfile.read(length))
            question = payload["question"]
            if not isinstance(question, str) or not question.strip() or len(question) > 1000:
                raise ValueError("question must be 1-1000 characters")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.send_error(400)
            return
        trace = answer(question, RunStore(ROOT / "examples" / "data"))
        body = json.dumps({"answer": trace.answer, "trace": {
            "mode": trace.mode, "tool_calls": trace.tool_calls,
            "latency_ms": trace.latency_ms, "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "estimated_cost_usd": trace.estimated_cost_usd,
        }}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
