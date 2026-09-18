from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from helpers import make_nodes, make_response, query_bundle, score_answer
from llama_index.postprocessor.jev import JevRerank
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient


def test_fail_open_returns_original_top_n(mocker: Any) -> None:
    nodes = make_nodes(["first", "second", "third"], scores=[0.9, 0.5, 0.1])
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        side_effect=TimeoutError("jev timeout"),
    )
    reranker = JevRerank(top_n=2, raise_on_error=False)
    result = reranker.postprocess_nodes(nodes, query_bundle=query_bundle())
    assert [n.node.get_content() for n in result] == ["first", "second"]
    assert [n.score for n in result] == [0.9, 0.5]


def test_raise_on_error_reraises(mocker: Any) -> None:
    nodes = make_nodes(["only"])
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        side_effect=RuntimeError("429 too many requests"),
    )
    reranker = JevRerank(top_n=1, raise_on_error=True)
    with pytest.raises(RuntimeError, match="429"):
        reranker.postprocess_nodes(nodes, query_bundle=query_bundle())


def test_single_failed_call_fail_opens_whole_pass(mocker: Any) -> None:
    nodes = make_nodes(
        ["keep-original-0", "keep-original-1", "keep-original-2"],
        scores=[0.3, 0.2, 0.1],
    )

    def flaky(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        if state["passage"] == "keep-original-1":
            raise RuntimeError("5xx from Jev")
        return make_response(relevance=score_answer(3.0))

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=flaky)
    reranker = JevRerank(top_n=2, raise_on_error=False)
    result = reranker.postprocess_nodes(nodes, query_bundle=query_bundle())

    # Original retrieval order, truncated — not a mix of Jev scores and retrieval.
    assert [n.node.get_content() for n in result] == [
        "keep-original-0",
        "keep-original-1",
    ]
    assert [n.score for n in result] == [0.3, 0.2]


def test_fail_open_never_returns_empty_on_error(mocker: Any) -> None:
    nodes = make_nodes(["a", "b"])
    mocker.patch.object(TypeSafeClient, "system_one", side_effect=RuntimeError("boom"))
    result = JevRerank(top_n=5, raise_on_error=False).postprocess_nodes(
        nodes, query_bundle=query_bundle()
    )
    assert len(result) == 2


@pytest.mark.asyncio
async def test_async_fail_open(mocker: Any) -> None:
    nodes = make_nodes(["a", "b", "c"], scores=[1.0, 0.5, 0.0])
    mocker.patch.object(
        AsyncTypeSafeClient,
        "system_one",
        new=AsyncMock(side_effect=RuntimeError("async 5xx")),
    )
    result = await JevRerank(top_n=2, raise_on_error=False).apostprocess_nodes(
        nodes, query_bundle=query_bundle()
    )
    assert [n.node.get_content() for n in result] == ["a", "b"]
