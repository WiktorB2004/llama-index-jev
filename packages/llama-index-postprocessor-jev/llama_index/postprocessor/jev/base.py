"""Jev reranker: one System One call per retrieved node."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.postprocessor.jev.utils import (
    build_relevance_question,
    get_answer,
    resolve_api_key,
)
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)


class JevRerank(BaseNodePostprocessor):
    """Rerank retrieved nodes with TypeSafe Jev.

    Each node is scored independently. State is exactly
    ``{"query": query_str, "passage": passage_text}`` so Jev never has to
    pick a winner among many passages whose question ids it cannot see.
    """

    model: str = Field(
        default="jev-latest",
        description="TypeSafe model id passed to the client and per call.",
    )
    top_n: int = Field(
        default=5,
        description="Number of nodes to return after sorting by Jev score.",
    )
    mode: Literal["noul", "score"] = Field(
        default="score",
        description=(
            "noul: binary relevance probability in 0–1. "
            "score: 4-level graded rubric in 0–3."
        ),
    )
    confidence_threshold: float | None = Field(
        default=None,
        description=(
            "Score mode only. Never drops nodes; sets metadata "
            "`jev_low_confidence` when answer.confidence is below this value."
        ),
    )
    timeout_s: float = Field(
        default=2.5,
        description="HTTP timeout in seconds, forwarded to TypeSafeClient.",
    )
    raise_on_error: bool = Field(
        default=False,
        description="If True, rerank errors raise. If False, fail open.",
    )
    max_concurrency: int = Field(
        default=8,
        description="Maximum in-flight system_one calls.",
    )

    _client: TypeSafeClient = PrivateAttr()
    _async_client: AsyncTypeSafeClient = PrivateAttr()

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        api_key = resolve_api_key(api_key)
        super().__init__(**kwargs)
        self._client = TypeSafeClient(
            api_key=api_key, model=self.model, timeout=self.timeout_s
        )
        self._async_client = AsyncTypeSafeClient(
            api_key=api_key, model=self.model, timeout=self.timeout_s
        )

    @classmethod
    def class_name(cls) -> str:
        return "JevRerank"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            dispatcher.event(ReRankEndEvent(nodes=[]))
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            try:
                scored = self._score_nodes(nodes, query_bundle.query_str)
                reranked = sorted(scored, key=lambda n: n.score or 0.0, reverse=True)[
                    : self.top_n
                ]
            except Exception:
                if self.raise_on_error:
                    raise
                logger.warning(
                    "Jev rerank failed; returning original nodes truncated to top_n",
                    exc_info=True,
                )
                reranked = nodes[: self.top_n]
            event.on_end(payload={EventPayload.NODES: reranked})

        dispatcher.event(ReRankEndEvent(nodes=reranked))
        return reranked

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            dispatcher.event(ReRankEndEvent(nodes=[]))
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            try:
                scored = await self._ascore_nodes(nodes, query_bundle.query_str)
                reranked = sorted(scored, key=lambda n: n.score or 0.0, reverse=True)[
                    : self.top_n
                ]
            except Exception:
                if self.raise_on_error:
                    raise
                logger.warning(
                    "Jev rerank failed; returning original nodes truncated to top_n",
                    exc_info=True,
                )
                reranked = nodes[: self.top_n]
            event.on_end(payload={EventPayload.NODES: reranked})

        dispatcher.event(ReRankEndEvent(nodes=reranked))
        return reranked

    def _score_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        # Collect every answer first. A single failed call aborts the whole
        # pass so we never mix Jev scores with original retrieval scores.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [pool.submit(self._score_one, node, query_str) for node in nodes]
            answers = [future.result() for future in futures]
        return [
            self._apply_answer(node, answer) for node, answer in zip(nodes, answers)
        ]

    async def _ascore_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def one(node: NodeWithScore) -> Any:
            async with semaphore:
                return await self._ascore_one(node, query_str)

        answers = await asyncio.gather(*[one(node) for node in nodes])
        return [
            self._apply_answer(node, answer) for node, answer in zip(nodes, answers)
        ]

    def _score_one(self, node: NodeWithScore, query_str: str) -> Any:
        response = self._client.system_one(
            state=self._state_for(node, query_str),
            questions={"relevance": build_relevance_question(self.mode)},
            model=self.model,
        )
        return get_answer(response, "relevance", self.mode)

    async def _ascore_one(self, node: NodeWithScore, query_str: str) -> Any:
        response = await self._async_client.system_one(
            state=self._state_for(node, query_str),
            questions={"relevance": build_relevance_question(self.mode)},
            model=self.model,
        )
        return get_answer(response, "relevance", self.mode)

    def _state_for(self, node: NodeWithScore, query_str: str) -> dict[str, str]:
        return {
            "query": query_str,
            "passage": node.node.get_content(metadata_mode=MetadataMode.EMBED),
        }

    def _apply_answer(self, node: NodeWithScore, answer: Any) -> NodeWithScore:
        node.node.metadata["retrieval_score"] = node.score
        if self.mode == "score":
            score = float(answer.score)
            node.node.metadata["jev_confidence"] = answer.confidence
            if (
                self.confidence_threshold is not None
                and answer.confidence < self.confidence_threshold
            ):
                node.node.metadata["jev_low_confidence"] = True
        else:
            score = float(answer.noul)
        return NodeWithScore(node=node.node, score=score)
