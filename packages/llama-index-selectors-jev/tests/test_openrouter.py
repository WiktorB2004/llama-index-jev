from __future__ import annotations

from typing import Any

import pytest
from helpers import DEFAULT_CHOICES, query_bundle
from llama_index.selectors.jev import JevMultiSelector, JevSingleSelector
from llama_index.selectors.jev.openrouter import (
    resolve_openrouter_model,
    wrap_response,
)
from llama_index.selectors.jev.utils import get_answer, resolve_api_key
from typesafe_sdk import TypeSafeClient


def test_resolve_openrouter_model() -> None:
    assert resolve_openrouter_model("jev-latest") == "~typesafe/jev-latest"


def test_openrouter_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        resolve_api_key(provider="openrouter")


def test_wrap_choice_answer() -> None:
    response = wrap_response(
        {
            "answers": {
                "route": {"type": "choice", "choice": "weather", "confidence": 0.91}
            }
        }
    )
    answer = get_answer(response, "route", "choice")
    assert answer.choice == "weather"
    assert answer.confidence == pytest.approx(0.91)


def test_openrouter_single_selector(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    typesafe = mocker.patch.object(TypeSafeClient, "system_one")

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert body["model"] == "~typesafe/jev-latest"
        assert body["state"] == {"query": "What is the weather in Paris?"}
        assert body["questions"]["route"]["type"] == "choice"
        return {
            "answers": {
                "route": {"type": "choice", "choice": "weather", "confidence": 0.88}
            }
        }

    mocker.patch(
        "llama_index.selectors.jev.openrouter.post_decisions",
        side_effect=fake,
    )
    result = JevSingleSelector(provider="openrouter").select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.ind == 0
    assert "weather" in result.reason
    typesafe.assert_not_called()


def test_openrouter_multi_selector(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        answers = {
            key: {"type": "noul", "noul": 0.9 if key == "weather" else 0.1}
            for key in body["questions"]
        }
        return {"answers": answers}

    mocker.patch(
        "llama_index.selectors.jev.openrouter.post_decisions",
        side_effect=fake,
    )
    result = JevMultiSelector(provider="openrouter", threshold=0.5).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.inds == [0]
