from __future__ import annotations

from typing import Any

import pytest
from helpers import (
    DEFAULT_CHOICES,
    choice_answer,
    make_response,
    query_bundle,
)
from llama_index.selectors.jev import JevMultiSelector, JevSingleSelector
from typesafe_sdk import TypeSafeClient


def test_single_selector_api_error_uses_default_index(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient, "system_one", side_effect=RuntimeError("429")
    )
    result = JevSingleSelector(default_index=1).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert len(result.selections) == 1
    assert result.ind == 1
    assert "fallback" in result.reason
    assert "default_index=1" in result.reason
    assert "429" in result.reason


def test_single_selector_api_error_raises_without_default(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient, "system_one", side_effect=RuntimeError("timeout")
    )
    with pytest.raises(RuntimeError, match="timeout"):
        JevSingleSelector().select(DEFAULT_CHOICES, query_bundle())


def test_multi_selector_api_error_uses_default_index(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient, "system_one", side_effect=RuntimeError("5xx")
    )
    result = JevMultiSelector(default_index=2).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.ind == 2
    assert "fallback" in result.reason
    assert "5xx" in result.reason


def test_multi_selector_api_error_raises_without_default(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient, "system_one", side_effect=RuntimeError("boom")
    )
    with pytest.raises(RuntimeError, match="boom"):
        JevMultiSelector().select(DEFAULT_CHOICES, query_bundle())


def test_out_of_schema_choice_triggers_fallback(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("not_a_tool", 0.99)),
    )
    result = JevSingleSelector(default_index=0).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.ind == 0
    assert "fallback" in result.reason
    assert "not_a_tool" in result.reason


def test_out_of_schema_choice_raises_without_default(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("ghost", 0.99)),
    )
    with pytest.raises(ValueError, match="ghost"):
        JevSingleSelector().select(DEFAULT_CHOICES, query_bundle())


def test_low_confidence_is_failure_with_fallback(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("weather", 0.2)),
    )
    result = JevSingleSelector(
        default_index=2, confidence_threshold=0.5
    ).select(DEFAULT_CHOICES, query_bundle())
    assert result.ind == 2
    assert "fallback" in result.reason
    assert "0.20" in result.reason


def test_low_confidence_raises_without_default(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("weather", 0.2)),
    )
    with pytest.raises(ValueError, match="confidence"):
        JevSingleSelector(confidence_threshold=0.5).select(
            DEFAULT_CHOICES, query_bundle()
        )


def test_in_schema_high_confidence_is_success(mocker: Any) -> None:
    mocker.patch.object(
        TypeSafeClient,
        "system_one",
        return_value=make_response(route=choice_answer("docs", 0.51)),
    )
    result = JevSingleSelector(confidence_threshold=0.5).select(
        DEFAULT_CHOICES, query_bundle()
    )
    assert result.ind == 2
    assert result.reason == "Jev selected 'docs' (confidence=0.51)"
