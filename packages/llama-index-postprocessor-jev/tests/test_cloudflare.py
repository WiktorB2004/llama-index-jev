from __future__ import annotations

from typing import Any

import pytest
from helpers import make_nodes, query_bundle
from llama_index.postprocessor.jev import JevRerank
from llama_index.postprocessor.jev.cloudflare import (
    resolve_cloudflare_model,
    run_url,
    unwrap_run_response,
    wrap_response,
)
from llama_index.postprocessor.jev.utils import get_answer, resolve_api_key
from typesafe_sdk import TypeSafeClient


def test_resolve_cloudflare_model() -> None:
    assert resolve_cloudflare_model("jev-latest") == "typesafe/jev"
    assert resolve_cloudflare_model("typesafe/jev") == "typesafe/jev"
    with pytest.raises(ValueError, match="typesafe/jev"):
        resolve_cloudflare_model("typesafe/jev-1.13")


def test_run_url_uses_account_path() -> None:
    assert (
        run_url("abc123")
        == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run"
    )


def test_cloudflare_token_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="CLOUDFLARE_API_TOKEN"):
        resolve_api_key(provider="cloudflare")


def test_cloudflare_explicit_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert resolve_api_key("cf-token", provider="cloudflare") == "cf-token"


def test_cloudflare_account_id_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        JevRerank(provider="cloudflare")


def test_unwrap_top_level_and_result_envelope() -> None:
    top_level = unwrap_run_response(
        {
            "model": "jev-1.13.0",
            "answers": {"relevance": {"type": "noul", "noul": 0.2}},
        }
    )
    enveloped = unwrap_run_response(
        {
            "success": True,
            "result": {
                "model": "jev-1.13.0",
                "answers": {"relevance": {"type": "noul", "noul": 0.7}},
            },
        }
    )
    assert get_answer(wrap_response(top_level), "relevance", "noul").noul == (
        pytest.approx(0.2)
    )
    assert get_answer(wrap_response(enveloped), "relevance", "noul").noul == (
        pytest.approx(0.7)
    )
    with pytest.raises(RuntimeError, match="Cloudflare"):
        unwrap_run_response({"success": False, "errors": [{"message": "nope"}]})


def test_cloudflare_rerank_posts_input_body(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    typesafe = mocker.patch.object(TypeSafeClient, "system_one")

    def fake(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "cf-token"
        assert kwargs["account_id"] == "acct-1"
        assert set(body) == {"model", "input"}
        assert body["model"] == "typesafe/jev"
        assert set(body["input"]["state"]) == {"query", "passage"}
        assert body["input"]["questions"]["relevance"]["type"] == "noul"
        noul = 0.95 if body["input"]["state"]["passage"] == "yes" else 0.1
        return {"answers": {"relevance": {"type": "noul", "noul": noul}}}

    mocker.patch(
        "llama_index.postprocessor.jev.cloudflare.post_run",
        side_effect=fake,
    )
    nodes = make_nodes(["no", "yes"], scores=[0.9, 0.1])
    result = JevRerank(provider="cloudflare", mode="noul", top_n=2).postprocess_nodes(
        nodes, query_bundle=query_bundle()
    )
    assert [n.node.get_content() for n in result] == ["yes", "no"]
    typesafe.assert_not_called()


def test_cloudflare_http_error_fail_open(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    mocker.patch(
        "llama_index.postprocessor.jev.cloudflare.post_run",
        side_effect=RuntimeError("Cloudflare AI run 429: rate limit"),
    )
    nodes = make_nodes(["a", "b"], scores=[0.8, 0.2])
    result = JevRerank(
        provider="cloudflare", top_n=1, raise_on_error=False
    ).postprocess_nodes(nodes, query_bundle=query_bundle())
    assert [n.node.get_content() for n in result] == ["a"]
