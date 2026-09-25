"""OpenRouter Decisions API transport for Jev (not TypeSafe /v1/systemone)."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from llama_index.selectors.jev.vercel import AsyncVercelClient, VercelClient
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"
API_KEY_ENV = "OPENROUTER_API_KEY"


def resolve_openrouter_model(model: str) -> str:
    """Map TypeSafe ids such as ``jev-latest`` to OpenRouter's ``~typesafe/...`` slug."""
    if model.startswith("~") or "/" in model:
        return model
    return f"~typesafe/{model}"


def serialize_question(question: Any) -> dict[str, Any]:
    if isinstance(question, Mapping):
        return dict(question)
    dump = getattr(question, "model_dump", None)
    if callable(dump):
        payload = dump()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError(f"Cannot serialize question of type {type(question)!r}")


def wrap_response(data: Mapping[str, Any]) -> SimpleNamespace:
    answers = data.get("answers") or {}
    if not isinstance(answers, Mapping):
        answers = {}
    return SimpleNamespace(
        model=data.get("model"),
        answers=dict(answers),
        usage=data.get("usage"),
        id=data.get("id"),
    )


def post_decisions(
    body: Mapping[str, Any],
    *,
    api_key: str,
    timeout: float,
    url: str = DECISIONS_URL,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/WiktorB2004/llama-index-jev",
        "X-OpenRouter-Title": "llama-index-jev",
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter decisions {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter decisions request failed: {exc}") from exc

    data = json.loads(raw)
    if not isinstance(data, dict) or "answers" not in data:
        raise RuntimeError(
            f"OpenRouter decisions response missing answers: {data!r}"[:500]
        )
    return data


class OpenRouterClient:
    """Sync client with the same ``system_one`` call shape as TypeSafeClient."""

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        timeout: float = 2.5,
    ) -> None:
        self.api_key = api_key
        self.model = resolve_openrouter_model(model)
        self.timeout = timeout

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        body = {
            "model": resolve_openrouter_model(model or self.model),
            "state": state,
            "questions": {
                key: serialize_question(question) for key, question in questions.items()
            },
        }
        return wrap_response(
            post_decisions(body, api_key=self.api_key, timeout=self.timeout)
        )


class AsyncOpenRouterClient:
    def __init__(self, sync_client: OpenRouterClient) -> None:
        self._sync = sync_client

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        return await asyncio.to_thread(
            self._sync.system_one, state, questions, model=model, **kwargs
        )


def make_clients(
    provider: str, api_key: str, model: str, timeout_s: float
) -> tuple[Any, Any]:
    if provider == "openrouter":
        sync = OpenRouterClient(api_key=api_key, model=model, timeout=timeout_s)
        return sync, AsyncOpenRouterClient(sync)
    if provider == "vercel":
        return (
            VercelClient(api_key=api_key, model=model, timeout=timeout_s),
            AsyncVercelClient(api_key=api_key, model=model, timeout=timeout_s),
        )
    if provider != "typesafe":
        raise ValueError(
            f"Unknown provider {provider!r}; "
            "expected 'typesafe', 'openrouter', or 'vercel'"
        )
    return (
        TypeSafeClient(api_key=api_key, model=model, timeout=timeout_s),
        AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout_s),
    )
