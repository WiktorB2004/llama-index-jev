from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def score_answer(score: float, confidence: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(score=score, confidence=confidence)


def noul_answer(noul: float) -> SimpleNamespace:
    return SimpleNamespace(noul=noul)


def make_response(**answers: Any) -> SimpleNamespace:
    return SimpleNamespace(answers=answers)


def make_nodes(
    texts: Sequence[str], scores: Iterable[float] | None = None
) -> list[NodeWithScore]:
    score_list = list(scores) if scores is not None else [0.1] * len(texts)
    return [
        NodeWithScore(
            node=TextNode(text=text, id_=str(i), metadata={}),
            score=score,
        )
        for i, (text, score) in enumerate(zip(texts, score_list))
    ]


def query_bundle(text: str = "Where is the Eiffel Tower?") -> QueryBundle:
    return QueryBundle(query_str=text)
