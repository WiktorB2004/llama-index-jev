from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from helpers import DEFAULT_CHOICES, query_bundle
from llama_index.selectors.jev import JevMultiSelector, JevSingleSelector
from llama_index.selectors.jev.utils import resolve_api_key
from llama_index.selectors.jev.vercel import VERCEL_BASE_URL, resolve_vercel_model
from typesafe_sdk import TypeSafeClient


def test_resolve_vercel_model() -> None:
    assert resolve_vercel_model("jev-latest") == "typesafe-ai/jev"
    assert resolve_vercel_model("typesafe-ai/jev") == "typesafe-ai/jev"


def test_vercel_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
        resolve_api_key(provider="vercel")


def test_vercel_single_selector(mocker: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    constructed: list[dict[str, Any]] = []
    original = TypeSafeClient.__init__

    def spy(self: TypeSafeClient, **kwargs: Any) -> None:
        constructed.append(kwargs)
        original(self, **kwargs)

    mocker.patch.object(TypeSafeClient, "__init__", spy)
    decisions = mocker.patch("llama_index.selectors.jev.openrouter.post_decisions")

    def fake(state: dict[str, Any], questions: dict[str, Any], **kwargs: Any) -> Any:
        assert kwargs["model"] == "typesafe-ai/jev"
        assert state == {"query": "What is the weather in Paris?"}
        assert questions["route"].type == "choice"
        return SimpleNamespace(
            answers={
                "route": {"type": "choice", "choice": "weather", "confidence": 0.88}
            }
        )

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    result = JevSingleSelector(provider="vercel").select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.ind == 0
    assert "weather" in result.reason
    assert constructed[0]["base_url"] == VERCEL_BASE_URL
    assert constructed[0]["api_key"] == "gw-key"
    decisions.assert_not_called()


def test_vercel_multi_selector(mocker: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")

    def fake(state: dict[str, Any], questions: dict[str, Any], **kwargs: Any) -> Any:
        assert kwargs["model"] == "typesafe-ai/jev"
        answers = {
            key: {"type": "noul", "noul": 0.9 if key == "weather" else 0.1}
            for key in questions
        }
        return SimpleNamespace(answers=answers)

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    result = JevMultiSelector(provider="vercel", threshold=0.5).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.inds == [0]
