from __future__ import annotations

from typing import Any

from helpers import make_nodes, make_response, query_bundle, score_answer
from llama_index.postprocessor.jev import JevRerank
from typesafe_sdk import TypeSafeClient


def test_one_system_one_call_per_node(mocker: Any) -> None:
    n = 20
    nodes = make_nodes([f"passage {i}" for i in range(n)])
    mock = mocker.patch.object(
        TypeSafeClient,
        "system_one",
        side_effect=[
            make_response(relevance=score_answer(float(i % 4))) for i in range(n)
        ],
    )
    reranker = JevRerank(top_n=5, max_concurrency=4)
    result = reranker.postprocess_nodes(nodes, query_bundle=query_bundle())

    assert mock.call_count == n
    assert len(result) == 5
    for call in mock.call_args_list:
        state = call.kwargs["state"]
        assert set(state.keys()) == {"query", "passage"}
        assert "passages" not in state
        assert isinstance(state["passage"], str)
        # Never more than one passage in state.
        assert not isinstance(state["passage"], (list, dict))
