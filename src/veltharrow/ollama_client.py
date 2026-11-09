"""Minimal HTTP client for the local Ollama server.

This is the only inference backend wired into the benchmark. It speaks Ollama's REST
API (``/api/tags``, ``/api/pull``, ``/api/embed``, ``/api/chat``) using nothing but the
standard library, so the core scripts stay dependency-light. The client picks a
reachable host at construction time (preferred host plus optional fallbacks) and
exposes the few calls the pipeline needs: list/pull models, embed text, and chat.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error, request

from veltharrow.config import (
    FINALIZE_OPTIONS,
    GENERATION_OPTIONS,
    OLLAMA_FALLBACK_HOSTS,
    OLLAMA_HOST,
    THINKING_GENERATION_OPTIONS,
)


class OllamaError(RuntimeError):
    """Raised for any failed Ollama request (HTTP error, unreachable host, or an
    error event in a streamed response).

    ``status`` carries the HTTP status code when the server answered with one, and is
    ``None`` when the request never got a response (unreachable host, timeout). Callers
    classify on this attribute rather than by substring-matching the message: a 5xx or a
    ``None`` status is a *backend fault* — the server broke — while a 4xx is a bad
    request from us. Neither is the model declining to answer.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _normalize_host(raw: str) -> str:
    """Add a default ``http://`` scheme if missing and strip a trailing slash."""
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


class OllamaClient:
    """Thin synchronous wrapper over the Ollama REST API.

    On construction it resolves the host to use: an explicit ``host`` argument or the
    ``OLLAMA_HOST`` env var takes precedence over the config default, and any
    ``fallback_hosts`` are tried in order until one answers ``/api/tags``.
    """

    def __init__(
        self,
        host: str | None = None,
        timeout: int = 600,
        fallback_hosts: tuple[str, ...] | None = None,
    ) -> None:
        """Resolve and store the host to talk to: ``host``/``OLLAMA_HOST`` wins over the
        config default, then each fallback is probed in order until one answers."""
        explicit_host = host or os.environ.get("OLLAMA_HOST")
        preferred = explicit_host or OLLAMA_HOST
        candidates: list[str] = [_normalize_host(preferred)]
        for fb in fallback_hosts if fallback_hosts is not None else OLLAMA_FALLBACK_HOSTS:
            normalized = _normalize_host(fb)
            if normalized not in candidates:
                candidates.append(normalized)
        self.timeout = timeout
        self.host = self._pick_reachable_host(candidates)

    @staticmethod
    def _pick_reachable_host(candidates: list[str]) -> str:
        """Return the first candidate host that answers ``/api/tags``, or raise
        :class:`OllamaError` listing everything tried if none respond."""
        last_error: Exception | None = None
        for host in candidates:
            try:
                with request.urlopen(f"{host}/api/tags", timeout=5) as resp:
                    resp.read()
                print(f"Ollama reachable at {host}", flush=True)
                return host
            except (error.URLError, OSError) as exc:
                print(f"Ollama not reachable at {host}: {exc}", flush=True)
                last_error = exc
        raise OllamaError(
            f"No reachable Ollama host. Tried {candidates}. Last error: {last_error!r}"
        )

    def _post(self, path: str, payload: dict[str, Any], timeout: int | None = None) -> Any:
        """POST ``payload`` as JSON to ``path`` and return the decoded response
        (or ``None`` for an empty body). Wraps HTTP/URL errors in :class:`OllamaError`."""
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = f"{exc}"
            if body:
                detail = f"{detail}: {body}"
            raise OllamaError(
                f"Ollama request failed for {path}: {detail}", status=exc.code
            ) from exc
        except error.URLError as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc
        if not body:
            return None
        return json.loads(body)

    def _get(self, path: str, timeout: int | None = None) -> Any:
        """GET ``path`` and return the decoded JSON response."""
        try:
            with request.urlopen(f"{self.host}{path}", timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc

    def list_models(self) -> set[str]:
        """Return the set of installed model tags (names from ``/api/tags``)."""
        data = self._get("/api/tags", timeout=20)
        return {item["name"] for item in data.get("models", [])}

    def pull_model(self, model: str) -> None:
        """Pull ``model`` into Ollama, streaming progress and raising
        :class:`OllamaError` if the server reports an error event."""
        req = request.Request(
            f"{self.host}/api/pull",
            data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=None) as resp:
                for raw_line in resp:
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line.decode("utf-8"))
                    if "error" in event:
                        raise OllamaError(f"Failed to pull {model}: {event['error']}")
        except error.URLError as exc:
            raise OllamaError(f"Failed to pull {model}: {exc}") from exc

    def embed(self, model: str, texts: list[str], keep_alive: int = 0) -> list[list[float]]:
        """Embed ``texts`` with ``model``, returning one vector per input.

        Tries the batch ``/api/embed`` endpoint first and falls back to per-text
        ``/api/embeddings`` calls for older servers that lack it. ``keep_alive=0``
        unloads the embedder immediately so it doesn't hold VRAM during inference.
        """
        if not texts:
            return []
        try:
            data = self._post("/api/embed", {"model": model, "input": texts, "keep_alive": keep_alive}, timeout=900)
            embeddings = data.get("embeddings")
            if embeddings is not None:
                return embeddings
        except OllamaError:
            pass

        embeddings = []
        for text in texts:
            data = self._post("/api/embeddings", {"model": model, "prompt": text, "keep_alive": keep_alive}, timeout=900)
            embedding = data.get("embedding")
            if embedding is None:
                raise OllamaError(f"Embedding response from {model} did not include an embedding")
            embeddings.append(embedding)
        return embeddings

    def chat(
        self,
        model: str,
        prompt: str,
        think: bool,
        *,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        finalize: bool = True,
    ) -> dict[str, Any]:
        """Run one chat completion and return content, thinking, latency and token counts.

        With ``think=True`` and ``finalize=True`` (the MCQ path), if the model emits
        only reasoning and no answer, a second non-thinking "finalize" call coerces a
        single-letter answer out of it. Free-form generation overrides ``system`` and
        ``options`` and passes ``finalize=False`` to skip that MCQ-specific fallback.
        """
        # Defaults reproduce the MCQ path exactly. Free-form generation passes its
        # own `system` prompt, a larger `options` budget, and finalize=False to skip
        # the single-letter MCQ fallback.
        if system is None:
            system = (
                "You answer multiple-choice questions. Return exactly one uppercase "
                "letter from A, B, C, D, E, or F. Do not include explanation."
            )
        if options is None:
            options = THINKING_GENERATION_OPTIONS if think else GENERATION_OPTIONS
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": think,
            "keep_alive": 3600,
            "options": options,
        }
        started = time.perf_counter()
        try:
            data = self._post("/api/chat", payload, timeout=900)
        except OllamaError:
            if not think:
                raise
            payload.pop("think", None)
            data = self._post("/api/chat", payload, timeout=900)
        elapsed = time.perf_counter() - started
        message = data.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")
        fallback_used = False
        fallback_latency = 0.0
        fallback_eval_count = ""
        fallback_prompt_eval_count = ""

        if finalize and think and not content.strip() and thinking:
            finalize_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a multiple-choice answer formatter. Return exactly one "
                            "uppercase letter from A, B, C, D, E, or F. No other text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": FINALIZE_OPTIONS,
            }
            fallback_started = time.perf_counter()
            fallback_data = self._post("/api/chat", finalize_payload, timeout=120)
            fallback_latency = time.perf_counter() - fallback_started
            fallback_message = fallback_data.get("message", {})
            content = fallback_message.get("content", "")
            fallback_used = True
            fallback_prompt_eval_count = fallback_data.get("prompt_eval_count", "")
            fallback_eval_count = fallback_data.get("eval_count", "")

        return {
            "content": content,
            "thinking": thinking,
            "latency_seconds": elapsed + fallback_latency,
            "prompt_eval_count": data.get("prompt_eval_count", ""),
            "eval_count": data.get("eval_count", ""),
            "fallback_used": fallback_used,
            "fallback_prompt_eval_count": fallback_prompt_eval_count,
            "fallback_eval_count": fallback_eval_count,
        }
