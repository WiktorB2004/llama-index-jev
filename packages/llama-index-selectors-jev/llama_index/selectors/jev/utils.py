"""Choice-key assignment and question constructors for Jev selectors."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar, overload

from llama_index.core.bridge.pydantic import BaseModel, ConfigDict
from llama_index.core.tools.types import ToolMetadata
from typesafe_sdk import Choice, Noul

MAX_QUESTIONS_PER_CALL = 255


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
    if provider == "vercel":
        key = os.environ.get("AI_GATEWAY_API_KEY")
        if not key:
            raise ValueError(
                "Vercel AI Gateway API key is missing. Pass api_key=... or set "
                "the AI_GATEWAY_API_KEY environment variable."
            )
        return key
    if provider == "cloudflare":
        key = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not key:
            raise ValueError(
                "Cloudflare API token is missing. Pass api_key=... or set the "
                "CLOUDFLARE_API_TOKEN environment variable."
            )
        return key
    if provider != "typesafe":
        raise ValueError(
            f"Unknown provider {provider!r}; "
            "expected 'typesafe', 'openrouter', 'vercel', or 'cloudflare'"
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


def assign_choice_keys(
    choices: Sequence[ToolMetadata],
) -> tuple[list[str], dict[str, int]]:
    """Map each choice to a stable key the model will see.

    Use `choice.name` when it is a non-empty string unique among the choices;
    otherwise `option_{i}`.
    """
    counts: dict[str, int] = {}
    for choice in choices:
        name = choice.name
        if name:
            counts[name] = counts.get(name, 0) + 1

    keys: list[str] = []
    key_to_index: dict[str, int] = {}
    for i, choice in enumerate(choices):
        name = choice.name
        if name and counts.get(name, 0) == 1:
            key = name
        else:
            key = f"option_{i}"
        keys.append(key)
        key_to_index[key] = i
    return keys, key_to_index


def choice_description(choice: ToolMetadata, key: str) -> str:
    return choice.description or choice.name or key


def build_route_question(
    choices: Sequence[ToolMetadata], keys: Sequence[str]
) -> Choice:
    criteria = {
        key: choice_description(choice, key) for key, choice in zip(keys, choices)
    }
    return Choice(
        instructions="Which option best answers this query?",
        criteria=criteria,
    )


def build_noul_questions(
    choices: Sequence[ToolMetadata], keys: Sequence[str]
) -> dict[str, Noul]:
    return {
        key: Noul(
            instructions="Is this option relevant to the query?",
            criteria={
                "true": choice_description(choice, key),
                "false": "Not relevant to the query",
            },
        )
        for key, choice in zip(keys, choices)
    }


def chunk_questions(
    questions: Mapping[str, Any],
    size: int = MAX_QUESTIONS_PER_CALL,
) -> list[dict[str, Any]]:
    """Split a question map into chunks of at most `size` keys."""
    items = list(questions.items())
    if not items:
        return []
    return [dict(items[i : i + size]) for i in range(0, len(items), size)]
