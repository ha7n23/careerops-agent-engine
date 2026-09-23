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
        (ChatModelProfile.STRUCTURED_OUTPUT, "minimal"),
        (ChatModelProfile.TOOL_CALLING, "low"),
    ],
)
def test_create_google_chat_model_uses_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
    profile: ChatModelProfile,
    expected_thinking_level: str,
) -> None:
    """Google construction should preserve the configured model behaviour."""

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
