from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from helpers import make_nodes, query_bundle
from llama_index.postprocessor.jev import JevRerank
from llama_index.postprocessor.jev.utils import resolve_api_key
from llama_index.postprocessor.jev.vercel import VERCEL_BASE_URL, resolve_vercel_model
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient


def test_resolve_vercel_model() -> None:
    assert resolve_vercel_model("jev-latest") == "typesafe-ai/jev"
    assert resolve_vercel_model("jev-1.13.0") == "typesafe-ai/jev"
    assert resolve_vercel_model("typesafe-ai/jev") == "typesafe-ai/jev"


def test_vercel_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
        resolve_api_key(provider="vercel")


def test_vercel_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    assert resolve_api_key("gw-key", provider="vercel") == "gw-key"


def test_vercel_rerank_uses_gateway_client(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    constructed: list[dict[str, Any]] = []
    original_sync = TypeSafeClient.__init__
    original_async = AsyncTypeSafeClient.__init__

    def spy_sync(self: TypeSafeClient, **kwargs: Any) -> None:
        constructed.append(kwargs)
        original_sync(self, **kwargs)

    def spy_async(self: AsyncTypeSafeClient, **kwargs: Any) -> None:
        constructed.append(kwargs)
        original_async(self, **kwargs)

    mocker.patch.object(TypeSafeClient, "__init__", spy_sync)
    mocker.patch.object(AsyncTypeSafeClient, "__init__", spy_async)
    decisions = mocker.patch("llama_index.postprocessor.jev.openrouter.post_decisions")

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        assert kwargs["model"] == "typesafe-ai/jev"
        assert questions["relevance"].type == "noul"
        assert set(state.keys()) == {"query", "passage"}
        noul = 0.95 if state["passage"] == "yes" else 0.1
        return SimpleNamespace(answers={"relevance": {"type": "noul", "noul": noul}})

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    nodes = make_nodes(["no", "yes"], scores=[0.9, 0.1])
    result = JevRerank(provider="vercel", mode="noul", top_n=2).postprocess_nodes(
        nodes, query_bundle=query_bundle()
    )
    assert [n.node.get_content() for n in result] == ["yes", "no"]
    assert len(constructed) == 2
    for kwargs in constructed:
        assert kwargs["api_key"] == "gw-key"
        assert kwargs["base_url"] == VERCEL_BASE_URL
        assert kwargs["model"] == "typesafe-ai/jev"
    decisions.assert_not_called()


def test_vercel_passthrough_model_slug(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        assert kwargs["model"] == "typesafe-ai/jev"
        return SimpleNamespace(answers={"relevance": {"type": "noul", "noul": 0.4}})

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    nodes = make_nodes(["a"])
    JevRerank(provider="vercel", model="typesafe-ai/jev", top_n=1).postprocess_nodes(
        nodes, query_bundle=query_bundle()
    )
