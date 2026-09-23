"""Provider-neutral construction of chat models."""

from enum import StrEnum
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from careerops_agent_engine.core.config import Settings


class ChatModelProfile(StrEnum):
    """Supported model behaviour profiles."""

    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"


def create_chat_model(
    *,
    settings: Settings,
    profile: ChatModelProfile,
    rate_limiter: BaseRateLimiter | None = None,
) -> BaseChatModel:
    """Create the configured provider's chat model."""

    if settings.llm_provider == "google":
        thinking_level: Literal["minimal", "low"] = (
            "low" if profile is ChatModelProfile.TOOL_CALLING else "minimal"
        )

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            thinking_level=thinking_level,
            rate_limiter=rate_limiter,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
