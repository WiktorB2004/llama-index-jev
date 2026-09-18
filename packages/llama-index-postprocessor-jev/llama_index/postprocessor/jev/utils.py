"""Question text and small helpers for the Jev reranker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal, TypeVar, overload

from llama_index.core.bridge.pydantic import BaseModel, ConfigDict
from typesafe_sdk import Noul, Score

SCORE_CRITERIA = [
    "Off-topic; does not relate to the query",
    "Tangentially related; mentions the topic but does not answer the query",
    "Partly answers; relevant but incomplete or missing key details",
    "Fully answers the query",
]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    noul: float
    type: Literal["noul"] | None = None


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: float
    confidence: float | None = None
    type: Literal["score"] | None = None


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choice: str
    confidence: float | None = None
    type: Literal["choice"] | None = None


_KIND_TO_ATTR = {
    "noul": "nouls",
    "score": "scores",
    "choice": "choices",
}

_TAnswer = TypeVar("_TAnswer", NoulAnswer, ScoreAnswer, ChoiceAnswer)


def _validate_answer(model: type[_TAnswer], raw: Any) -> _TAnswer:
    parsed = model.model_validate(raw, from_attributes=True)
    if not isinstance(parsed, model):
        raise TypeError(f"expected {model.__name__}")
    return parsed


def resolve_api_key(api_key: str | None = None, *, provider: str = "typesafe") -> str:
    """Return an explicit key or the env var for ``provider``."""
    if api_key:
        return api_key
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key is missing. Pass api_key=... or set the "
                "OPENROUTER_API_KEY environment variable."
            )
        return key
    if provider != "typesafe":
        raise ValueError(
            f"Unknown provider {provider!r}; expected 'typesafe' or 'openrouter'"
        )
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError(
            "TypeSafe API key is missing. Pass api_key=... or set the "
            "TYPESAFE_API_KEY environment variable."
        )
    return key


def _raw_answer(response: Any, key: str, kind: str) -> Any:
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and key in answers:
        return answers[key]
    attr = _KIND_TO_ATTR[kind]
    return getattr(response, attr)[key]


@overload
def get_answer(response: Any, key: str, kind: Literal["noul"]) -> NoulAnswer: ...


@overload
def get_answer(response: Any, key: str, kind: Literal["score"]) -> ScoreAnswer: ...


@overload
def get_answer(response: Any, key: str, kind: Literal["choice"]) -> ChoiceAnswer: ...


def get_answer(
    response: Any, key: str, kind: str
) -> NoulAnswer | ScoreAnswer | ChoiceAnswer:
    """Prefer `response.answers[key]`; fall back to `.nouls` / `.scores` / `.choices`."""
    raw = _raw_answer(response, key, kind)
    if kind == "noul":
        return _validate_answer(NoulAnswer, raw)
    if kind == "score":
        return _validate_answer(ScoreAnswer, raw)
    if kind == "choice":
        return _validate_answer(ChoiceAnswer, raw)
    raise ValueError(f"Unknown answer kind {kind!r}")


def build_relevance_question(mode: str) -> Noul | Score:
    """Build the single per-passage relevance question for `mode`."""
    if mode == "noul":
        return Noul(
            instructions="Does this passage answer the query?",
            criteria={
                "true": "The passage contains information that answers the query",
                "false": "The passage is off-topic or only loosely related",
            },
        )
    return Score(
        instructions="How well does this passage answer the query?",
        criteria=SCORE_CRITERIA,
    )
