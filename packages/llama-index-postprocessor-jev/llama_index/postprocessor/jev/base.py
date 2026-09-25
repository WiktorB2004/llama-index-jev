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
from llama_index.postprocessor.jev.openrouter import make_clients
from llama_index.postprocessor.jev.utils import (
    NoulAnswer,
    ScoreAnswer,
    build_relevance_question,
    get_answer,
    resolve_api_key,
)

logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)


class JevRerank(BaseNodePostprocessor):
    """Rerank retrieved nodes with TypeSafe Jev.

    Each node is scored independently. State is exactly
    ``{"query": query_str, "passage": passage_text}`` so Jev never has to
    pick a winner among many passages whose question ids it cannot see.
    """

    provider: Literal["typesafe", "openrouter", "vercel", "cloudflare"] = Field(
        default="typesafe",
        description=(
            "typesafe: TypeSafe System One SDK. "
            "openrouter: OpenRouter Decisions API (OPENROUTER_API_KEY). "
            "vercel: Vercel AI Gateway TypeSafe API (AI_GATEWAY_API_KEY). "
            "cloudflare: Cloudflare Workers AI (CLOUDFLARE_API_TOKEN, "
            "CLOUDFLARE_ACCOUNT_ID)."
        ),
    )
    model: str = Field(
        default="jev-latest",
        description=(
            "TypeSafe model id. Remapped to ~typesafe/... on OpenRouter. "
            "On Vercel, an id with no slash is sent as typesafe-ai/jev. "
            "On Cloudflare, jev-latest and typesafe/jev are sent as typesafe/jev."
        ),
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
        description="HTTP timeout in seconds, forwarded to the Jev client.",
    )
    raise_on_error: bool = Field(
        default=False,
        description="If True, rerank errors raise. If False, fail open.",
    )
    max_concurrency: int = Field(
        default=8,
        description="Maximum in-flight system_one calls.",
    )

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Create a Jev reranker.

        Args:
            api_key: TypeSafe, OpenRouter, Vercel, or Cloudflare credential.
                Falls back to ``TYPESAFE_API_KEY``, ``OPENROUTER_API_KEY``,
                ``AI_GATEWAY_API_KEY``, or ``CLOUDFLARE_API_TOKEN`` based on
                ``provider``. Cloudflare also requires ``CLOUDFLARE_ACCOUNT_ID``.
            **kwargs: Fields such as ``provider``, ``top_n``, and ``mode``.
        """
        super().__init__(**kwargs)
        api_key = resolve_api_key(api_key, provider=self.provider)
        self._client, self._async_client = make_clients(
            self.provider, api_key, self.model, self.timeout_s
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

        async def one(node: NodeWithScore) -> NoulAnswer | ScoreAnswer:
            async with semaphore:
                return await self._ascore_one(node, query_str)

        answers = await asyncio.gather(*[one(node) for node in nodes])
        return [
            self._apply_answer(node, answer) for node, answer in zip(nodes, answers)
        ]

    def _score_one(
        self, node: NodeWithScore, query_str: str
    ) -> NoulAnswer | ScoreAnswer:
        response = self._client.system_one(
            state=self._state_for(node, query_str),
            questions={"relevance": build_relevance_question(self.mode)},
            model=self.model,
        )
        return get_answer(response, "relevance", self.mode)

    async def _ascore_one(
        self, node: NodeWithScore, query_str: str
    ) -> NoulAnswer | ScoreAnswer:
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

    def _apply_answer(
        self, node: NodeWithScore, answer: NoulAnswer | ScoreAnswer
    ) -> NodeWithScore:
        node.node.metadata["retrieval_score"] = node.score
        if self.mode == "score":
            if not isinstance(answer, ScoreAnswer):
                raise TypeError(
                    f"score mode expected ScoreAnswer, got {type(answer).__name__}"
                )
            score = float(answer.score)
            if answer.confidence is not None:
                node.node.metadata["jev_confidence"] = answer.confidence
                if (
                    self.confidence_threshold is not None
                    and answer.confidence < self.confidence_threshold
                ):
                    node.node.metadata["jev_low_confidence"] = True
        else:
            if not isinstance(answer, NoulAnswer):
                raise TypeError(
                    f"noul mode expected NoulAnswer, got {type(answer).__name__}"
                )
            score = float(answer.noul)
        return NodeWithScore(node=node.node, score=score)
