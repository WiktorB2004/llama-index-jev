"""Vercel AI Gateway transport for Jev (TypeSafe SDK, gateway base URL)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

VERCEL_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
VERCEL_MODEL = "typesafe-ai/jev"
API_KEY_ENV = "AI_GATEWAY_API_KEY"


def resolve_vercel_model(model: str) -> str:
    """Map a bare TypeSafe id such as ``jev-latest`` to ``typesafe-ai/jev``.

    A slug that already contains ``/`` is sent unchanged.
    """
    if "/" in model:
        return model
    return VERCEL_MODEL


class VercelClient:
    """Sync client. Remaps model ids before the TypeSafe SDK sees them.

    Callers pass the public id (``jev-latest``) on every ``system_one`` call.
    Vercel expects ``typesafe-ai/jev``, so remapping has to happen here and not
    only as the client default.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        timeout: float = 2.5,
    ) -> None:
        self.model = resolve_vercel_model(model)
        self._client = TypeSafeClient(
            api_key=api_key,
            model=self.model,
            timeout=timeout,
            base_url=VERCEL_BASE_URL,
        )

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._client.system_one(
            state,
            questions,
            model=resolve_vercel_model(model or self.model),
            **kwargs,
        )


class AsyncVercelClient:
    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        timeout: float = 2.5,
    ) -> None:
        self.model = resolve_vercel_model(model)
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=self.model,
            timeout=timeout,
            base_url=VERCEL_BASE_URL,
        )

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._client.system_one(
            state,
            questions,
            model=resolve_vercel_model(model or self.model),
            **kwargs,
        )
