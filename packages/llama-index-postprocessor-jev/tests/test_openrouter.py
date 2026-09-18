from __future__ import annotations

from typing import Any

import pytest
from helpers import make_nodes, query_bundle
from llama_index.postprocessor.jev import JevRerank
from llama_index.postprocessor.jev.openrouter import (
    resolve_openrouter_model,
    serialize_question,
    wrap_response,
)
from llama_index.postprocessor.jev.utils import get_answer, resolve_api_key
from typesafe_sdk import Noul, TypeSafeClient


def test_resolve_openrouter_model() -> None:
    assert resolve_openrouter_model("jev-latest") == "~typesafe/jev-latest"
    assert resolve_openrouter_model("~typesafe/jev-latest") == "~typesafe/jev-latest"
    assert (
        resolve_openrouter_model("typesafe/jev-1.13-20260917")
        == "typesafe/jev-1.13-20260917"
    )


def test_openrouter_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        resolve_api_key(provider="openrouter")


def test_wrap_response_matches_get_answer() -> None:
    response = wrap_response(
        {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {"relevance": {"type": "noul", "noul": 0.97}},
        }
    )
    assert get_answer(response, "relevance", "noul").noul == pytest.approx(0.97)


def test_serialize_noul() -> None:
    payload = serialize_question(
        Noul(instructions="Does this passage answer the query?")
    )
    assert payload["type"] == "noul"
    assert "instructions" in payload


def test_openrouter_rerank_does_not_call_typesafe(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    typesafe = mocker.patch.object(TypeSafeClient, "system_one")

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "or-key"
        assert body["model"] == "~typesafe/jev-latest"
        assert set(body["state"].keys()) == {"query", "passage"}
        assert body["questions"]["relevance"]["type"] == "noul"
        noul = 0.95 if body["state"]["passage"] == "yes" else 0.1
        return {"answers": {"relevance": {"type": "noul", "noul": noul}}}

    mocker.patch(
        "llama_index.postprocessor.jev.openrouter.post_decisions",
        side_effect=fake,
    )
    nodes = make_nodes(["no", "yes"], scores=[0.9, 0.1])
    result = JevRerank(provider="openrouter", mode="noul", top_n=2).postprocess_nodes(
        nodes, query_bundle=query_bundle()
    )
    assert [n.node.get_content() for n in result] == ["yes", "no"]
    typesafe.assert_not_called()


def test_openrouter_http_error_fail_open(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    mocker.patch(
        "llama_index.postprocessor.jev.openrouter.post_decisions",
        side_effect=RuntimeError("OpenRouter decisions 429: rate limit"),
    )
    nodes = make_nodes(["a", "b"], scores=[0.8, 0.2])
    result = JevRerank(
        provider="openrouter", top_n=1, raise_on_error=False
    ).postprocess_nodes(nodes, query_bundle=query_bundle())
    assert [n.node.get_content() for n in result] == ["a"]
