from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from helpers import (
    DEFAULT_CHOICES,
    choice_answer,
    make_response,
    noul_answer,
    query_bundle,
    tools,
)
from llama_index.core.tools.types import ToolMetadata
from llama_index.selectors.jev import JevMultiSelector, JevSingleSelector
from llama_index.selectors.jev.utils import (
    MAX_QUESTIONS_PER_CALL,
    chunk_questions,
    get_answer,
)
from typesafe_sdk import AsyncTypeSafeClient, Choice, TypeSafeClient


def test_single_selector_returns_index_and_reason(mocker: Any) -> None:
    mock = mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("weather", 0.95)),
    )
    selector = JevSingleSelector()
    result = selector.select(DEFAULT_CHOICES, query_bundle())

    assert len(result.selections) == 1
    assert result.ind == 0
    assert result.reason == "Jev selected 'weather' (confidence=0.95)"

    questions = mock.call_args.kwargs["questions"]
    criteria = questions["route"].criteria
    assert "weather" in criteria
    assert "option_0" not in criteria
    assert mock.call_args.kwargs["state"] == {"query": "What is the weather in Paris?"}


def test_single_selector_uses_unique_choice_names_as_keys(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("docs", 0.80)),
    )
    result = JevSingleSelector().select(DEFAULT_CHOICES, query_bundle())
    assert result.ind == 2
    assert "docs" in result.reason


def test_single_selector_falls_back_to_option_i_when_names_collide(
    mocker: Any,
) -> None:
    choices = [
        ToolMetadata(name="dup", description="first"),
        ToolMetadata(name="dup", description="second"),
    ]
    mock = mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("option_1", 0.70)),
    )
    result = JevSingleSelector().select(choices, query_bundle())
    assert result.ind == 1
    criteria = mock.call_args.kwargs["questions"]["route"].criteria
    assert set(criteria) == {"option_0", "option_1"}


def test_multi_selector_returns_above_threshold_in_original_order(
    mocker: Any,
) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(
            weather=noul_answer(0.9),
            sports=noul_answer(0.2),
            docs=noul_answer(0.7),
        ),
    )
    result = JevMultiSelector(threshold=0.5).select(DEFAULT_CHOICES, query_bundle())
    assert result.inds == [0, 2]
    assert result.reasons == [
        "Jev noul=0.90 for 'weather'",
        "Jev noul=0.70 for 'docs'",
    ]


def test_multi_selector_empty_threshold_returns_highest_noul(
    mocker: Any,
) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(
            weather=noul_answer(0.2),
            sports=noul_answer(0.4),
            docs=noul_answer(0.1),
        ),
    )
    result = JevMultiSelector(threshold=0.5).select(DEFAULT_CHOICES, query_bundle())
    assert len(result.selections) == 1
    assert result.ind == 1
    assert result.reason == "Jev noul=0.40 for 'sports'"


def test_multi_selector_uses_strict_greater_than_threshold(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(
            weather=noul_answer(0.5),
            sports=noul_answer(0.51),
            docs=noul_answer(0.1),
        ),
    )
    result = JevMultiSelector(threshold=0.5).select(DEFAULT_CHOICES, query_bundle())
    assert result.inds == [1]


def test_multi_selector_chunks_questions_above_255(mocker: Any) -> None:
    n = MAX_QUESTIONS_PER_CALL + 1
    choices = tools(*[(f"opt_{i}", f"desc {i}") for i in range(n)])

    def fake(state: dict[str, str], questions: dict[str, Any], **kwargs: Any) -> Any:
        assert len(questions) <= MAX_QUESTIONS_PER_CALL
        return make_response(**{key: noul_answer(0.9) for key in questions})

    mock = mocker.patch.object(TypeSafeClient, "system_one", side_effect=fake)
    result = JevMultiSelector(threshold=0.5).select(choices, query_bundle())
    assert mock.call_count == 2
    assert len(result.selections) == n


def test_chunk_questions_helper() -> None:
    questions = {f"k{i}": i for i in range(256)}
    chunks = chunk_questions(questions, size=255)
    assert [len(c) for c in chunks] == [255, 1]


def test_class_names() -> None:
    assert JevSingleSelector.class_name() == "JevSingleSelector"
    assert JevMultiSelector.class_name() == "JevMultiSelector"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevSingleSelector()


def test_route_question_is_choice(mocker: Any) -> None:
    mock = mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("weather")),
    )
    JevSingleSelector().select(DEFAULT_CHOICES, query_bundle())
    question = mock.call_args.kwargs["questions"]["route"]
    assert isinstance(question, Choice)


@pytest.mark.asyncio
async def test_async_single_selector(mocker: Any) -> None:
    mocker.patch.object(
        AsyncTypeSafeClient,
        "system_one",
        new=AsyncMock(return_value=make_response(route=choice_answer("sports", 0.88))),
    )
    result = await JevSingleSelector().aselect(DEFAULT_CHOICES, query_bundle())
    assert result.ind == 1
    assert result.reason == "Jev selected 'sports' (confidence=0.88)"


@pytest.mark.asyncio
async def test_async_multi_selector(mocker: Any) -> None:
    mocker.patch.object(
        AsyncTypeSafeClient,
        "system_one",
        new=AsyncMock(
            return_value=make_response(
                weather=noul_answer(0.1),
                sports=noul_answer(0.8),
                docs=noul_answer(0.9),
            )
        ),
    )
    result = await JevMultiSelector(threshold=0.5).aselect(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.inds == [1, 2]


def test_get_answer_rejects_invalid_choice() -> None:
    from llama_index.core.bridge.pydantic import ValidationError

    response = make_response(route=noul_answer(0.5))
    with pytest.raises(ValidationError):
        get_answer(response, "route", "choice")
