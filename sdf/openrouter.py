"""Minimal OpenRouter chat client with a disk cache and bounded retries (question generation, grading)."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1/chat/completions"


def api_key(env_file: str = ".env") -> str:
    """OPENROUTER_API_KEY from the environment, else from a .env file."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    path = pathlib.Path(env_file)
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("Set OPENROUTER_API_KEY (environment or .env)")


class Client:
    def __init__(self, key: str, cache: pathlib.Path):
        self.key, self.cache = key, cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.spend = {"prompt": 0, "completion": 0, "calls": 0, "cached": 0}

    def chat(self, model: str, prompt: str, max_tokens: int = 1200, no_reasoning: bool = False) -> str:
        digest = hashlib.sha256(f"{model}\x00{max_tokens}\x00{no_reasoning}\x00{prompt}".encode()).hexdigest()
        hit = self.cache / f"{digest}.json"
        if hit.exists():
            with self.lock:
                self.spend["cached"] += 1
            return json.loads(hit.read_text())["text"]
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0}
        if no_reasoning:  # reasoning tokens are billed as output and are not needed here
            payload["reasoning"] = {"enabled": False}
        last = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                text = body["choices"][0]["message"].get("content") or ""
                if not text.strip():
                    raise RuntimeError("empty completion")
                usage = body.get("usage", {})
                with self.lock:
                    self.spend["calls"] += 1
                    self.spend["prompt"] += usage.get("prompt_tokens", 0)
                    self.spend["completion"] += usage.get("completion_tokens", 0)
                hit.write_text(json.dumps({"model": model, "text": text}))
                return text
            except (urllib.error.URLError, RuntimeError, KeyError, TimeoutError) as error:
                last = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last}")
