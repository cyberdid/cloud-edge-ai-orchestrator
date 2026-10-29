"""OpenAI-compatible router: one endpoint for OpenEvolve, two backends behind it.

Each call is routed to Gemma (local Ollama) or Argon (``agy``) by a seeded RNG keyed on
the call index, so the arm's model mix is reproducible and resumable. Every call is
logged. A backend failure returns an empty reply (a failed mutation): it is never
retried and never replaced by the other model.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Literal

Backend = Callable[[list[dict]], tuple]
ARGON_MODEL = "gemini-4-argon-medium"
QUOTA_PATTERN = re.compile(r"quota|rate.?limit|resource.?exhausted|exhaust|\b429\b", re.IGNORECASE)
PAUSE_FILE = "QUOTA_PAUSED"
# agy is an agent; in headless mode a tool call is denied and the turn ends empty
# (probe 2026-10-29: 1 of 3 calls). Ask for a direct text answer.
AGY_NO_TOOLS = ("Answer directly in text. Do not call any tools, run commands or read files: you have no "
                "workspace, and everything you need is below. Reply only with the requested SEARCH/REPLACE "
                "blocks and a short explanation.\n\n")


def choose_model(seed: int, call_index: int, cloud_share: float) -> Literal["gemma", "argon"]:
    return "argon" if random.Random(f"{seed}:{call_index}").random() < cloud_share else "gemma"


class Router:
    def __init__(self, seed: int, cloud_share: float, log_path: Path, *,
                 gemma: Backend, argon: Backend, start_index: int = 0, budget: int | None = None) -> None:
        self.seed, self.cloud_share, self.budget = seed, cloud_share, budget
        self.log_path = Path(log_path)
        self.backends = {"gemma": gemma, "argon": argon}
        self.call_index = start_index
        self.paused = False
        self.lock = threading.Lock()
        self.last_usage = (0, 0)

    def complete(self, messages: list[dict]) -> str:
        with self.lock:
            if self.paused or (self.budget is not None and self.call_index >= self.budget):
                self.last_usage = (0, 0)
                return ""  # budget spent or quota paused: not a model call, not logged
            index = self.call_index
            model = choose_model(self.seed, index, self.cloud_share)
            started, error = time.time(), ""
            try:
                reply = self.backends[model](messages)
                text, prompt_tokens, completion_tokens = reply[:3]
                status = reply[3] if len(reply) > 3 else ("ok" if text.strip() else "empty")
            except Exception as exc:  # a failed call is a failed mutation, not a crash
                if model == "argon" and QUOTA_PATTERN.search(str(exc)):
                    # quota is the environment, not the model: pause, keep the slot for resume
                    self.paused = True
                    self.last_usage = (0, 0)
                    (self.log_path.parent / PAUSE_FILE).write_text(json.dumps(
                        {"call_index": index, "ts": started, "error": str(exc)[:500]}))
                    return ""
                text, prompt_tokens, completion_tokens = "", 0, 0
                status, error = f"error:{type(exc).__name__}", str(exc)[:500]
            self.call_index += 1
            self.last_usage = (prompt_tokens, completion_tokens)
            record = {
                "call_index": index, "model": model, "ts_start": started,
                "latency_s": time.time() - started, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "status": status, "error": error,
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(), "response": text,
            }
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            return text


def serve(router: Router, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text = router.complete(body.get("messages", []))
            prompt_tokens, completion_tokens = router.last_usage
            payload = json.dumps({
                "id": f"pilot-{router.call_index}", "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "pilot-router"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def ollama_gemma(base_url: str = "http://localhost:11434", model: str = "gemma4:12b", num_predict: int = 2048,
                 timeout_s: float = 900, temperature: float = 0.7, top_p: float = 0.95) -> Backend:
    def backend(messages: list[dict]) -> tuple:
        body = json.dumps({
            "model": model, "messages": messages, "stream": False, "think": False,
            "options": {"num_predict": num_predict, "temperature": temperature, "top_p": top_p},
        }).encode()
        request = urllib.request.Request(f"{base_url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(request, timeout=timeout_s))
        text = reply.get("message", {}).get("content", "")
        status = "truncated" if reply.get("done_reason") == "length" else ("ok" if text.strip() else "empty")
        return text, int(reply.get("prompt_eval_count", 0)), int(reply.get("eval_count", 0)), status

    return backend


def agy_argon(model: str = ARGON_MODEL, timeout_s: float = 1200) -> Backend:
    from antigravity_hybrid_orchestrator.architect.client import CloudArchitectClient

    client = CloudArchitectClient(backend="agy", model=model, timeout=timeout_s)

    def backend(messages: list[dict]) -> tuple:
        prompt = "\n\n".join(f"{m.get('role', 'user')}:\n{m.get('content', '')}" for m in messages)
        return client.complete(AGY_NO_TOOLS + prompt)

    return backend
