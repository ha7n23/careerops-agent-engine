"""Tests for provider-neutral structured-output fallback construction."""

from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from careerops_agent_engine.infrastructure.llm.structured_output import (
    create_structured_output_runnable,
)


class _TemporaryProviderError(Exception):
    """Represent a provider error that permits fallback."""


def _raise_temporary_error(_value: object) -> dict[str, str]:
    """Simulate temporary failure of the primary model."""

    raise _TemporaryProviderError


def _raise_invalid_request(_value: object) -> dict[str, str]:
    """Simulate a non-fallback request failure."""

    raise ValueError("invalid request")


def test_returns_primary_structured_runnable_without_fallback() -> None:
    """A missing fallback should preserve the primary runnable."""

    model = Mock(spec=BaseChatModel)
    primary = RunnableLambda(lambda _value: {"model": "primary"})
    model.with_structured_output.return_value = primary

    structured = create_structured_output_runnable(
        model=model,
        schema={"type": "object"},
    )

    assert structured is primary


def test_falls_back_for_explicit_temporary_exception() -> None:
    """An allowed temporary failure should invoke the fallback once."""

    model = Mock(spec=BaseChatModel)
    fallback_model = Mock(spec=BaseChatModel)

    model.with_structured_output.return_value = RunnableLambda(_raise_temporary_error)
    fallback_model.with_structured_output.return_value = RunnableLambda(
        lambda _value: {"model": "fallback"}
    )

    structured = create_structured_output_runnable(
        model=model,
        schema={"type": "object"},
        fallback_model=fallback_model,
        fallback_exceptions=(_TemporaryProviderError,),
    )

    assert structured.invoke("input") == {"model": "fallback"}


def test_does_not_fallback_for_unlisted_exception() -> None:
    """Invalid requests must propagate without invoking the fallback."""

    model = Mock(spec=BaseChatModel)
    fallback_model = Mock(spec=BaseChatModel)
    fallback = Mock(return_value={"model": "fallback"})

    model.with_structured_output.return_value = RunnableLambda(_raise_invalid_request)
    fallback_model.with_structured_output.return_value = RunnableLambda(fallback)

    structured = create_structured_output_runnable(
        model=model,
        schema={"type": "object"},
        fallback_model=fallback_model,
        fallback_exceptions=(_TemporaryProviderError,),
    )

    with pytest.raises(ValueError, match="invalid request"):
        structured.invoke("input")

    fallback.assert_not_called()


def test_rejects_fallback_without_explicit_exceptions() -> None:
    """Fallback must never default to catching every exception."""

    model = Mock(spec=BaseChatModel)
    fallback_model = Mock(spec=BaseChatModel)

    model.with_structured_output.return_value = RunnableLambda(
        lambda _value: {"model": "primary"}
    )

    with pytest.raises(
        ValueError,
        match="requires explicit exception types",
    ):
        create_structured_output_runnable(
            model=model,
            schema={"type": "object"},
            fallback_model=fallback_model,
        )
