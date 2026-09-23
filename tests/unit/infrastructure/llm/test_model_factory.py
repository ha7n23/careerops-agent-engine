"""Tests for provider-neutral chat-model construction."""

from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter

from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.infrastructure.llm import model_factory
from careerops_agent_engine.infrastructure.llm.model_factory import (
    ChatModelProfile,
)


@pytest.mark.parametrize(
    ("profile", "expected_thinking_level"),
    [
        (
            ChatModelProfile.FAST_STRUCTURED_OUTPUT,
            "minimal",
        ),
        (
            ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
            "minimal",
        ),
        (
            ChatModelProfile.TOOL_CALLING,
            "low",
        ),
    ],
)
def test_create_google_chat_model_uses_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
    profile: ChatModelProfile,
    expected_thinking_level: str,
) -> None:
    """Google construction should preserve the configured behaviour."""

    fake_model = Mock(spec=BaseChatModel)
    constructor = Mock(return_value=fake_model)
    rate_limiter = Mock(spec=BaseRateLimiter)

    monkeypatch.setattr(
        model_factory,
        "ChatGoogleGenerativeAI",
        constructor,
    )

    settings = Settings(
        llm_provider="google",
        llm_model="gemini-test",
        llm_temperature=0.25,
        llm_timeout_seconds=12.0,
        llm_max_retries=1,
    )

    created_model = model_factory.create_chat_model(
        settings=settings,
        profile=profile,
        rate_limiter=rate_limiter,
    )

    assert created_model is fake_model
    constructor.assert_called_once_with(
        model="gemini-test",
        temperature=0.25,
        timeout=12.0,
        max_retries=1,
        thinking_level=expected_thinking_level,
        rate_limiter=rate_limiter,
    )


def test_create_groq_chat_model_uses_configured_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq construction should preserve neutral model settings."""

    fake_model = Mock(spec=BaseChatModel)
    constructor = Mock(return_value=fake_model)
    rate_limiter = Mock(spec=BaseRateLimiter)

    monkeypatch.setattr(
        model_factory,
        "ChatGroq",
        constructor,
    )

    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        llm_temperature=0.5,
        llm_timeout_seconds=15.0,
        llm_max_retries=1,
    )

    created_model = model_factory.create_chat_model(
        settings=settings,
        profile=ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
        rate_limiter=rate_limiter,
    )

    assert created_model is fake_model
    constructor.assert_called_once_with(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        timeout=15.0,
        max_retries=1,
        rate_limiter=rate_limiter,
    )


@pytest.mark.parametrize(
    ("profile", "expected_model"),
    [
        (
            ChatModelProfile.FAST_STRUCTURED_OUTPUT,
            "fast-model",
        ),
        (
            ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
            "quality-model",
        ),
        (
            ChatModelProfile.TOOL_CALLING,
            "tool-model",
        ),
    ],
)
def test_resolve_model_name_uses_profile_override(
    profile: ChatModelProfile,
    expected_model: str,
) -> None:
    """Each task profile should resolve its configured model."""

    settings = Settings(
        llm_model="default-model",
        llm_fast_model="fast-model",
        llm_quality_model="quality-model",
        llm_tool_model="tool-model",
    )

    assert model_factory.resolve_model_name(settings, profile) == expected_model


@pytest.mark.parametrize(
    "profile",
    [
        ChatModelProfile.FAST_STRUCTURED_OUTPUT,
        ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
        ChatModelProfile.TOOL_CALLING,
    ],
)
def test_resolve_model_name_falls_back_to_default(
    profile: ChatModelProfile,
) -> None:
    """Unset profile overrides should preserve existing configuration."""

    settings = Settings(
        llm_model="default-model",
    )

    assert model_factory.resolve_model_name(settings, profile) == "default-model"
