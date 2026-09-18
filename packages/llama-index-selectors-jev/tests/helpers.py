from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata


def choice_answer(choice: str, confidence: float = 0.95) -> SimpleNamespace:
    return SimpleNamespace(choice=choice, confidence=confidence)


def noul_answer(noul: float) -> SimpleNamespace:
    return SimpleNamespace(noul=noul)


def make_response(**answers: Any) -> SimpleNamespace:
    return SimpleNamespace(answers=answers)


def tools(*specs: tuple[str, str]) -> list[ToolMetadata]:
    return [ToolMetadata(name=name, description=desc) for name, desc in specs]


def query_bundle(text: str = "What is the weather in Paris?") -> QueryBundle:
    return QueryBundle(query_str=text)


DEFAULT_CHOICES: Sequence[ToolMetadata] = tools(
    ("weather", "Answers questions about weather and forecasts"),
    ("sports", "Answers questions about sports scores"),
    ("docs", "Answers questions about product documentation"),
)
