from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from helpers import (
    make_nodes,
    make_response,
    noul_answer,
    query_bundle,
    score_answer,
)
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.jev import JevRerank
from llama_index.postprocessor.jev.utils import get_answer
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient


def _state_from_call(call: Any) -> dict[str, Any]:
    if call.kwargs.get("state") is not None:
        return call.kwargs["state"]
    return call.args[0]


def _assert_one_pair_state(mock: Any, expected_query: str) -> None:
    assert mock.call_count >= 1
    for call in mock.call_args_list:
        state = _state_from_call(call)
        assert set(state.keys()) == {"query", "passage"}
        assert isinstance(state["query"], str)
        assert state["query"] == expected_query
        assert not isinstance(state["query"], QueryBundle)
        assert "passages" not in state


def test_score_mode_reorders_and_truncates(mocker: Any) -> None:
    nodes = make_nodes(
        ["off topic", "partial answer", "full answer"],
        scores=[0.9, 0.5, 0.1],
    )
    scores_by_passage = {
        "off topic": 0.2,
        "partial answer": 1.5,
        "full answer": 2.8,
    }

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        return make_response(
            relevance=score_answer(scores_by_passage[state["passage"]])
        )

    mock = mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    reranker = JevRerank(top_n=2, mode="score")
    query = query_bundle()
    result = reranker.postprocess_nodes(nodes, query_bundle=query)

    assert [n.node.get_content() for n in result] == ["full answer", "partial answer"]
    assert result[0].score == pytest.approx(2.8)
    assert result[1].score == pytest.approx(1.5)
    assert len(result) == 2
    _assert_one_pair_state(mock, query.query_str)
    assert mock.call_count == 3


def test_noul_mode_reorders(mocker: Any) -> None:
    nodes = make_nodes(["no", "yes"], scores=[0.8, 0.2])
    nouls_by_passage = {"no": 0.1, "yes": 0.95}

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        return make_response(relevance=noul_answer(nouls_by_passage[state["passage"]]))

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    reranker = JevRerank(top_n=2, mode="noul")
    result = reranker.postprocess_nodes(nodes, query_bundle=query_bundle())
    assert [n.node.get_content() for n in result] == ["yes", "no"]
    assert result[0].score == pytest.approx(0.95)


def test_score_mode_writes_confidence_metadata(mocker: Any) -> None:
    nodes = make_nodes(["low conf", "high conf"])

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        if state["passage"] == "low conf":
            return make_response(relevance=score_answer(2.0, confidence=0.2))
        return make_response(relevance=score_answer(1.0, confidence=0.9))

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    reranker = JevRerank(top_n=2, mode="score", confidence_threshold=0.5)
    result = reranker.postprocess_nodes(nodes, query_bundle=query_bundle())
    # Sorted by score: low conf (2.0) first, high conf (1.0) second.
    # confidence_threshold flags, it does not drop.
    assert len(result) == 2
    low, high = result
    assert low.node.metadata["jev_confidence"] == pytest.approx(0.2)
    assert low.node.metadata["jev_low_confidence"] is True
    assert high.node.metadata["jev_confidence"] == pytest.approx(0.9)
    assert "jev_low_confidence" not in high.node.metadata
    assert "retrieval_score" in low.node.metadata


def test_noul_mode_does_not_write_confidence_metadata(mocker: Any) -> None:
    nodes = make_nodes(["a", "b"])

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        return make_response(
            relevance=noul_answer(0.2 if state["passage"] == "a" else 0.8)
        )

    mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    reranker = JevRerank(top_n=2, mode="noul", confidence_threshold=0.5)
    result = reranker.postprocess_nodes(nodes, query_bundle=query_bundle())
    for node in result:
        assert "jev_confidence" not in node.node.metadata
        assert "jev_low_confidence" not in node.node.metadata
        assert "retrieval_score" in node.node.metadata


def test_query_in_state_is_string_not_query_bundle(mocker: Any) -> None:
    nodes = make_nodes(["only"])
    mock = mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(relevance=score_answer(2.0)),
    )
    query = QueryBundle(query_str="a real question")
    JevRerank(top_n=1).postprocess_nodes(nodes, query_bundle=query)
    state = _state_from_call(mock.call_args)
    assert state["query"] == "a real question"
    assert type(state["query"]) is str


def test_empty_nodes_short_circuit(mocker: Any) -> None:
    mock = mocker.patch.object(TypeSafeClient, "system_one")
    result = JevRerank().postprocess_nodes([], query_bundle=query_bundle())
    assert result == []
    mock.assert_not_called()


def test_missing_query_bundle_raises(mocker: Any) -> None:
    mocker.patch.object(TypeSafeClient, "system_one")
    with pytest.raises(ValueError, match="query bundle"):
        JevRerank().postprocess_nodes(make_nodes(["x"]), query_bundle=None)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevRerank()


def test_class_name() -> None:
    assert JevRerank.class_name() == "JevRerank"


def test_get_answer_falls_back_to_typed_map() -> None:
    answer = score_answer(1.5)
    response = type("Resp", (), {"scores": {"relevance": answer}})()
    parsed = get_answer(response, "relevance", "score")
    assert parsed.score == pytest.approx(1.5)
    assert parsed.confidence == pytest.approx(0.9)


def test_get_answer_rejects_invalid_payload() -> None:
    from llama_index.core.bridge.pydantic import ValidationError

    response = make_response(relevance=SimpleNamespace(noul="not-a-float"))
    with pytest.raises(ValidationError):
        get_answer(response, "relevance", "noul")


@pytest.mark.asyncio
async def test_async_score_mode(mocker: Any) -> None:
    nodes = make_nodes(["b", "a"], scores=[0.9, 0.1])
    scores_by_passage = {"b": 0.5, "a": 3.0}

    async def fake(
        state: dict[str, str], questions: dict[str, Any], **kwargs: Any
    ) -> Any:
        return make_response(
            relevance=score_answer(scores_by_passage[state["passage"]])
        )

    mocker.patch.object(
        AsyncTypeSafeClient, "system_one", new=AsyncMock(side_effect=fake)
    )
    reranker = JevRerank(top_n=2, mode="score")
    result = await reranker.apostprocess_nodes(nodes, query_bundle=query_bundle())
    assert [n.node.get_content() for n in result] == ["a", "b"]
    assert result[0].score == pytest.approx(3.0)
