"""Cloudflare Workers AI transport for Jev (not TypeSafe /v1/systemone)."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

RUN_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
WIRE_MODEL = "typesafe/jev"
API_KEY_ENV = "CLOUDFLARE_API_TOKEN"
ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
ACCEPTED_MODELS = frozenset({"jev-latest", WIRE_MODEL})


def resolve_cloudflare_model(model: str) -> str:
    """Map ``jev-latest`` and ``typesafe/jev`` to the Workers AI model id."""
    if model in ACCEPTED_MODELS:
        return WIRE_MODEL
    raise ValueError(
        f"Unsupported Cloudflare model {model!r}; "
        "expected 'jev-latest' or 'typesafe/jev'"
    )


def resolve_account_id(account_id: str | None = None) -> str:
    """Return an explicit account id or ``CLOUDFLARE_ACCOUNT_ID``."""
    account_id = account_id or os.environ.get(ACCOUNT_ID_ENV)
    if not account_id:
        raise ValueError(
            "Cloudflare account id is missing. Set the CLOUDFLARE_ACCOUNT_ID "
            "environment variable."
        )
    return account_id


def run_url(account_id: str) -> str:
    return RUN_URL.format(account_id=account_id)


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


def unwrap_run_response(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the decision object from either envelope Cloudflare documents.

    The model page shows ``answers`` at the top level. The Workers AI REST API
    also wraps that object under ``result``.
    """
    if data.get("success") is False:
        raise RuntimeError(f"Cloudflare AI run failed: {data.get('errors')!r}"[:500])
    result = data.get("result")
    if isinstance(result, Mapping) and "answers" in result:
        return dict(result)
    return dict(data)


def post_run(
    body: Mapping[str, Any],
    *,
    api_key: str,
    account_id: str,
    timeout: float,
    url: str | None = None,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(
        url or run_url(account_id),
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare AI run {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cloudflare AI run request failed: {exc}") from exc

    data = json.loads(raw)
    decision = unwrap_run_response(data) if isinstance(data, dict) else {}
    if "answers" not in decision:
        raise RuntimeError(
            f"Cloudflare AI run response missing answers: {data!r}"[:500]
        )
    return decision


class CloudflareClient:
    """Sync client with the same ``system_one`` call shape as TypeSafeClient."""

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        timeout: float = 2.5,
    ) -> None:
        self.api_key = api_key
        self.model = resolve_cloudflare_model(model)
        self.account_id = resolve_account_id()
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
            "model": resolve_cloudflare_model(model or self.model),
            "input": {
                "state": state,
                "questions": {
                    key: serialize_question(question)
                    for key, question in questions.items()
                },
            },
        }
        return wrap_response(
            post_run(
                body,
                api_key=self.api_key,
                account_id=self.account_id,
                timeout=self.timeout,
            )
        )


class AsyncCloudflareClient:
    def __init__(self, sync_client: CloudflareClient) -> None:
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
