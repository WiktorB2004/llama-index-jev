"""Jev selectors: Choice for single-route, Noul-per-option for multi-route."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from llama_index.core.base.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata
from llama_index.selectors.jev.utils import (
    assign_choice_keys,
    build_noul_questions,
    build_route_question,
    chunk_questions,
    get_answer,
    resolve_api_key,
)
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

logger = logging.getLogger(__name__)


def _fallback_or_raise(
    default_index: int | None, exc: BaseException
) -> SelectorResult:
    if default_index is None:
        raise exc
    logger.warning(
        "Jev selector failed; falling back to default_index=%s",
        default_index,
        exc_info=exc,
    )
    return SelectorResult(
        selections=[
            SingleSelection(
                index=default_index,
                reason=(
                    f"Jev fallback to default_index={default_index} "
                    f"after error: {type(exc).__name__}: {exc}"
                ),
            )
        ]
    )


class _JevSelectorBase(BaseSelector):
    """PromptMixin is abstract; Jev has no LLM prompts to get or set."""

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        return None


class JevSingleSelector(_JevSelectorBase):
    """Pick exactly one choice with a Jev Choice question.

    Fail-closed: an API error, an out-of-schema choice, or a confidence
    miss either raises or (if `default_index` is set) returns that index.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        default_index: int | None = None,
        confidence_threshold: float | None = None,
        timeout_s: float = 2.5,
    ) -> None:
        super().__init__()
        api_key = resolve_api_key(api_key)
        self.model = model
        self.default_index = default_index
        self.confidence_threshold = confidence_threshold
        self.timeout_s = timeout_s
        self._client = TypeSafeClient(
            api_key=api_key, model=model, timeout=timeout_s
        )
        self._async_client = AsyncTypeSafeClient(
            api_key=api_key, model=model, timeout=timeout_s
        )

    @classmethod
    def class_name(cls) -> str:
        return "JevSingleSelector"

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        try:
            return self._select_from_answer(
                self._call_system_one(choices, query.query_str), choices
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed on any Jev error
            return _fallback_or_raise(self.default_index, exc)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        try:
            return self._select_from_answer(
                await self._acall_system_one(choices, query.query_str),
                choices,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed on any Jev error
            return _fallback_or_raise(self.default_index, exc)

    def _call_system_one(
        self, choices: Sequence[ToolMetadata], query_str: str
    ) -> Any:
        keys, _ = assign_choice_keys(choices)
        return self._client.system_one(
            state={"query": query_str},
            questions={"route": build_route_question(choices, keys)},
            model=self.model,
        )

    async def _acall_system_one(
        self, choices: Sequence[ToolMetadata], query_str: str
    ) -> Any:
        keys, _ = assign_choice_keys(choices)
        return await self._async_client.system_one(
            state={"query": query_str},
            questions={"route": build_route_question(choices, keys)},
            model=self.model,
        )

    def _select_from_answer(
        self, response: Any, choices: Sequence[ToolMetadata]
    ) -> SelectorResult:
        _, key_to_index = assign_choice_keys(choices)
        answer = get_answer(response, "route", "choice")
        selected = answer.choice
        if selected not in key_to_index:
            raise ValueError(
                f"Jev returned choice '{selected}' which is not in "
                f"{sorted(key_to_index)}"
            )
        if (
            self.confidence_threshold is not None
            and answer.confidence < self.confidence_threshold
        ):
            raise ValueError(
                "Jev choice confidence "
                f"{answer.confidence:.2f} is below threshold "
                f"{self.confidence_threshold:.2f}"
            )
        return SelectorResult(
            selections=[
                SingleSelection(
                    index=key_to_index[selected],
                    reason=(
                        f"Jev selected '{selected}' "
                        f"(confidence={answer.confidence:.2f})"
                    ),
                )
            ]
        )


class JevMultiSelector(_JevSelectorBase):
    """Keep every choice whose Noul is above `threshold`.

    One `system_one` call carries one Noul per option (chunked at 255).
    If nothing clears the threshold, return the single highest-noul choice.
    `default_index` is used only when the API call itself fails.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        threshold: float = 0.5,
        default_index: int | None = None,
        timeout_s: float = 2.5,
    ) -> None:
        super().__init__()
        api_key = resolve_api_key(api_key)
        self.model = model
        self.threshold = threshold
        self.default_index = default_index
        self.timeout_s = timeout_s
        self._client = TypeSafeClient(
            api_key=api_key, model=model, timeout=timeout_s
        )
        self._async_client = AsyncTypeSafeClient(
            api_key=api_key, model=model, timeout=timeout_s
        )

    @classmethod
    def class_name(cls) -> str:
        return "JevMultiSelector"

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        try:
            nouls = self._collect_nouls(choices, query.query_str)
            return self._selections_from_nouls(choices, nouls)
        except Exception as exc:  # noqa: BLE001 - fail-closed on any Jev error
            return _fallback_or_raise(self.default_index, exc)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        try:
            nouls = await self._acollect_nouls(choices, query.query_str)
            return self._selections_from_nouls(choices, nouls)
        except Exception as exc:  # noqa: BLE001 - fail-closed on any Jev error
            return _fallback_or_raise(self.default_index, exc)

    def _collect_nouls(
        self, choices: Sequence[ToolMetadata], query_str: str
    ) -> dict[str, float]:
        keys, _ = assign_choice_keys(choices)
        questions = build_noul_questions(choices, keys)
        merged: dict[str, float] = {}
        for chunk in chunk_questions(questions):
            response = self._client.system_one(
                state={"query": query_str},
                questions=chunk,
                model=self.model,
            )
            for key in chunk:
                merged[key] = float(get_answer(response, key, "noul").noul)
        return merged

    async def _acollect_nouls(
        self, choices: Sequence[ToolMetadata], query_str: str
    ) -> dict[str, float]:
        keys, _ = assign_choice_keys(choices)
        questions = build_noul_questions(choices, keys)
        merged: dict[str, float] = {}
        for chunk in chunk_questions(questions):
            response = await self._async_client.system_one(
                state={"query": query_str},
                questions=chunk,
                model=self.model,
            )
            for key in chunk:
                merged[key] = float(get_answer(response, key, "noul").noul)
        return merged

    def _selections_from_nouls(
        self,
        choices: Sequence[ToolMetadata],
        nouls: dict[str, float],
    ) -> SelectorResult:
        keys, key_to_index = assign_choice_keys(choices)
        selected: list[SingleSelection] = []
        best_key: str | None = None
        best_noul = float("-inf")
        for key in keys:
            noul = nouls[key]
            if noul > best_noul:
                best_noul = noul
                best_key = key
            if noul > self.threshold:
                selected.append(
                    SingleSelection(
                        index=key_to_index[key],
                        reason=f"Jev noul={noul:.2f} for '{key}'",
                    )
                )
        if selected:
            return SelectorResult(selections=selected)
        if best_key is None:
            raise ValueError("Jev multi-selector received no choices.")
        return SelectorResult(
            selections=[
                SingleSelection(
                    index=key_to_index[best_key],
                    reason=f"Jev noul={best_noul:.2f} for '{best_key}'",
                )
            ]
        )
