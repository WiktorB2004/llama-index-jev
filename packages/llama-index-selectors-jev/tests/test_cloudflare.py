from __future__ import annotations

from typing import Any

import pytest
from helpers import DEFAULT_CHOICES, query_bundle
from llama_index.selectors.jev import JevMultiSelector, JevSingleSelector
from llama_index.selectors.jev.cloudflare import resolve_cloudflare_model
from llama_index.selectors.jev.utils import resolve_api_key
from typesafe_sdk import TypeSafeClient


def test_resolve_cloudflare_model() -> None:
    assert resolve_cloudflare_model("jev-latest") == "typesafe/jev"
    with pytest.raises(ValueError, match="typesafe/jev"):
        resolve_cloudflare_model("other")


def test_cloudflare_token_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="CLOUDFLARE_API_TOKEN"):
        resolve_api_key(provider="cloudflare")


def _cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")


def test_cloudflare_single_selector(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cloudflare_env(monkeypatch)
    typesafe = mocker.patch.object(TypeSafeClient, "system_one")

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["account_id"] == "acct-1"
        assert body["model"] == "typesafe/jev"
        assert body["input"]["state"] == {"query": "What is the weather in Paris?"}
        assert body["input"]["questions"]["route"]["type"] == "choice"
        return {
            "answers": {
                "route": {"type": "choice", "choice": "weather", "confidence": 0.88}
            }
        }

    mocker.patch("llama_index.selectors.jev.cloudflare.post_run", side_effect=fake)
    result = JevSingleSelector(provider="cloudflare").select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.ind == 0
    assert "weather" in result.reason
    typesafe.assert_not_called()


def test_cloudflare_multi_selector(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cloudflare_env(monkeypatch)

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert body["input"]["questions"]["weather"]["type"] == "noul"
        answers = {
            key: {"type": "noul", "noul": 0.9 if key == "weather" else 0.1}
            for key in body["input"]["questions"]
        }
        return {"answers": answers}

    mocker.patch("llama_index.selectors.jev.cloudflare.post_run", side_effect=fake)
    result = JevMultiSelector(provider="cloudflare", threshold=0.5).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.inds == [0]


def test_cloudflare_explicit_token_without_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "explicit-token"
        return {
            "answers": {
                "route": {"type": "choice", "choice": "docs", "confidence": 0.8}
            }
        }

    mocker.patch("llama_index.selectors.jev.cloudflare.post_run", side_effect=fake)
    result = JevSingleSelector(provider="cloudflare", api_key="explicit-token").select(
        DEFAULT_CHOICES, query_bundle("Where is the API reference?")
    )
    assert result.ind == 2
