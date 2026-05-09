from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 180

    def generate(self, prompt: str, temperature: float = 0.2, num_predict: int | None = None) -> str:
        options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": "30m",
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return str(body.get("response", "")).strip()

    def list_models(self) -> list[str]:
        req = request.Request(
            url=f"{self.base_url}/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return [str(model.get("name", "")).strip() for model in body.get("models", []) if model.get("name")]

    @classmethod
    def pick_available_model(
        cls,
        preferred: list[str] | None = None,
        base_url: str = "http://127.0.0.1:11434",
    ) -> str | None:
        preferred = preferred or ["llama3.1:8b", "llama3.1:latest", "gemma3:4b", "qwen3.5:latest"]
        probe = cls(model=preferred[0], base_url=base_url)
        try:
            available = set(probe.list_models())
        except Exception:
            return None
        for model in preferred:
            if model in available:
                return model
        return next(iter(available), None)
