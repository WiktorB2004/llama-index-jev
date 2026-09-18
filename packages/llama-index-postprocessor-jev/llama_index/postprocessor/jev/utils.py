"""Question text and small helpers for the Jev reranker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from typesafe_sdk import Noul, Score

SCORE_CRITERIA = [
    "Off-topic; does not relate to the query",
    "Tangentially related; mentions the topic but does not answer the query",
    "Partly answers; relevant but incomplete or missing key details",
    "Fully answers the query",
]

_KIND_TO_ATTR = {
    "noul": "nouls",
    "score": "scores",
    "choice": "choices",
}


def resolve_api_key(api_key: str | None = None) -> str:
    """Return an explicit key or `TYPESAFE_API_KEY`, else raise."""
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError(
            "TypeSafe API key is missing. Pass api_key=... or set the "
            "TYPESAFE_API_KEY environment variable."
        )
    return key


def get_answer(response: Any, key: str, kind: str) -> Any:
    """Prefer `response.answers[key]`; fall back to `.nouls` / `.scores` / `.choices`."""
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and key in answers:
        return answers[key]
    attr = _KIND_TO_ATTR[kind]
    return getattr(response, attr)[key]


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
