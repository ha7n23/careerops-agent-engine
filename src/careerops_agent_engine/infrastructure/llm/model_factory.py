"""Provider-neutral construction of chat models."""

from enum import StrEnum
from typing import Any, Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from careerops_agent_engine.core.config import Settings


class ChatModelProfile(StrEnum):
    """Supported model behaviour profiles."""

    FAST_STRUCTURED_OUTPUT = "fast_structured_output"
    QUALITY_STRUCTURED_OUTPUT = "quality_structured_output"
    TOOL_CALLING = "tool_calling"


def resolve_model_name(
    settings: Settings,
    profile: ChatModelProfile,
) -> str:
    """Resolve a profile-specific model or the default model."""

    if profile is ChatModelProfile.FAST_STRUCTURED_OUTPUT:
        return settings.llm_fast_model or settings.llm_model

    if profile is ChatModelProfile.QUALITY_STRUCTURED_OUTPUT:
        return settings.llm_quality_model or settings.llm_model

    if profile is ChatModelProfile.TOOL_CALLING:
        return settings.llm_tool_model or settings.llm_model

    return settings.llm_model


def create_chat_model(
    *,
    settings: Settings,
    profile: ChatModelProfile,
    rate_limiter: BaseRateLimiter | None = None,
) -> BaseChatModel:
    """Create the configured provider's chat model."""

    model_name = resolve_model_name(settings, profile)

    if settings.llm_provider == "google":
        thinking_level: Literal["minimal", "low"] = (
            "low" if profile is ChatModelProfile.TOOL_CALLING else "minimal"
        )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            thinking_level=thinking_level,
            rate_limiter=rate_limiter,
        )

    if settings.llm_provider == "groq":
        # ChatGroq exposes `model` as a Pydantic alias, but its generated
        # constructor signatures disagree between Pylance and mypy.
        # Contain that third-party typing mismatch at this provider boundary.
        groq_constructor = cast(Any, ChatGroq)

        return cast(
            BaseChatModel,
            groq_constructor(
                model=model_name,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                rate_limiter=rate_limiter,
            ),
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
