"""Tests for model-backed adapter factory wiring."""

from unittest.mock import Mock, call

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter

from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.infrastructure.llm import factory
from careerops_agent_engine.infrastructure.llm.model_factory import (
    ChatModelProfile,
)


class _TemporaryProviderError(Exception):
    """Represent a temporary provider-capacity failure."""


@pytest.mark.parametrize(
    (
        "factory_name",
        "adapter_name",
        "expected_profile",
        "expected_model",
    ),
    [
        (
            "create_requirement_extractor",
            "LangChainRequirementExtractor",
            ChatModelProfile.FAST_STRUCTURED_OUTPUT,
            "fast-model",
        ),
        (
            "create_cv_evidence_extractor",
            "LangChainCVEvidenceExtractor",
            ChatModelProfile.FAST_STRUCTURED_OUTPUT,
            "fast-model",
        ),
        (
            "create_cv_proposal_generator",
            "LangChainCVProposalGenerator",
            ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
            "quality-model",
        ),
        (
            "create_cv_claim_verifier",
            "LangChainCVClaimVerifier",
            ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
            "quality-model",
        ),
    ],
)
def test_structured_adapter_factory_uses_neutral_model(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    adapter_name: str,
    expected_profile: ChatModelProfile,
    expected_model: str,
) -> None:
    """Structured adapters should receive their configured model profile."""

    settings = Settings(
        llm_provider="google",
        llm_model="default-model",
        llm_fast_model="fast-model",
        llm_quality_model="quality-model",
        llm_tool_model="tool-model",
    )
    fake_model = Mock(spec=BaseChatModel)
    fake_adapter = object()
    rate_limiter = Mock(spec=BaseRateLimiter)

    limiter_factory = Mock(return_value=rate_limiter)
    model_factory = Mock(return_value=fake_model)
    adapter_factory = Mock(return_value=fake_adapter)

    monkeypatch.setattr(
        factory,
        "_get_rate_limiter",
        limiter_factory,
    )
    monkeypatch.setattr(
        factory,
        "create_chat_model",
        model_factory,
    )
    monkeypatch.setattr(
        factory,
        adapter_name,
        adapter_factory,
    )

    create_adapter = getattr(factory, factory_name)
    created_adapter = create_adapter(settings)

    assert created_adapter is fake_adapter
    limiter_factory.assert_called_once_with(
        settings,
        expected_profile,
    )
    model_factory.assert_called_once_with(
        settings=settings,
        profile=expected_profile,
        rate_limiter=rate_limiter,
    )

    if expected_profile is ChatModelProfile.FAST_STRUCTURED_OUTPUT:
        adapter_factory.assert_called_once_with(
            model=fake_model,
            model_name=expected_model,
            fallback_model=None,
            fallback_exceptions=(),
        )
    else:
        adapter_factory.assert_called_once_with(
            model=fake_model,
            model_name=expected_model,
        )


@pytest.mark.parametrize(
    ("factory_name", "adapter_name"),
    [
        (
            "create_requirement_extractor",
            "LangChainRequirementExtractor",
        ),
        (
            "create_cv_evidence_extractor",
            "LangChainCVEvidenceExtractor",
        ),
    ],
)
def test_groq_fast_structured_factory_adds_quality_fallback(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    adapter_name: str,
) -> None:
    """Groq fast structured adapters should receive one quality fallback."""

    settings = Settings(
        llm_provider="groq",
        llm_model="default-model",
        llm_fast_model="fast-model",
        llm_quality_model="quality-model",
        llm_tool_model="tool-model",
    )
    primary_model = Mock(spec=BaseChatModel)
    fallback_model = Mock(spec=BaseChatModel)
    fast_rate_limiter = Mock(spec=BaseRateLimiter)
    quality_rate_limiter = Mock(spec=BaseRateLimiter)
    fake_adapter = object()

    limiter_factory = Mock(
        side_effect=[
            fast_rate_limiter,
            quality_rate_limiter,
        ]
    )
    model_factory = Mock(
        side_effect=[
            primary_model,
            fallback_model,
        ]
    )
    adapter_factory = Mock(return_value=fake_adapter)
    exception_resolver = Mock(return_value=(_TemporaryProviderError,))

    monkeypatch.setattr(
        factory,
        "_get_rate_limiter",
        limiter_factory,
    )
    monkeypatch.setattr(
        factory,
        "create_chat_model",
        model_factory,
    )
    monkeypatch.setattr(
        factory,
        "resolve_transient_model_exceptions",
        exception_resolver,
    )
    monkeypatch.setattr(
        factory,
        adapter_name,
        adapter_factory,
    )

    create_adapter = getattr(factory, factory_name)
    created_adapter = create_adapter(settings)

    assert created_adapter is fake_adapter
    assert limiter_factory.call_args_list == [
        call(
            settings,
            ChatModelProfile.FAST_STRUCTURED_OUTPUT,
        ),
        call(
            settings,
            ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
        ),
    ]
    assert model_factory.call_args_list == [
        call(
            settings=settings,
            profile=ChatModelProfile.FAST_STRUCTURED_OUTPUT,
            rate_limiter=fast_rate_limiter,
        ),
        call(
            settings=settings,
            profile=ChatModelProfile.QUALITY_STRUCTURED_OUTPUT,
            rate_limiter=quality_rate_limiter,
        ),
    ]
    exception_resolver.assert_called_once_with(settings)
    adapter_factory.assert_called_once_with(
        model=primary_model,
        model_name="fast-model",
        fallback_model=fallback_model,
        fallback_exceptions=(_TemporaryProviderError,),
    )


def test_groq_fast_structured_factory_skips_same_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same model should not be constructed as its own fallback."""

    settings = Settings(
        llm_provider="groq",
        llm_model="shared-model",
        llm_fast_model="shared-model",
        llm_quality_model="shared-model",
        llm_tool_model="tool-model",
    )
    primary_model = Mock(spec=BaseChatModel)
    rate_limiter = Mock(spec=BaseRateLimiter)
    fake_adapter = object()

    limiter_factory = Mock(return_value=rate_limiter)
    model_factory = Mock(return_value=primary_model)
    adapter_factory = Mock(return_value=fake_adapter)
    exception_resolver = Mock(return_value=(_TemporaryProviderError,))

    monkeypatch.setattr(
        factory,
        "_get_rate_limiter",
        limiter_factory,
    )
    monkeypatch.setattr(
        factory,
        "create_chat_model",
        model_factory,
    )
    monkeypatch.setattr(
        factory,
        "resolve_transient_model_exceptions",
        exception_resolver,
    )
    monkeypatch.setattr(
        factory,
        "LangChainRequirementExtractor",
        adapter_factory,
    )

    created_adapter = factory.create_requirement_extractor(settings)

    assert created_adapter is fake_adapter
    limiter_factory.assert_called_once_with(
        settings,
        ChatModelProfile.FAST_STRUCTURED_OUTPUT,
    )
    model_factory.assert_called_once_with(
        settings=settings,
        profile=ChatModelProfile.FAST_STRUCTURED_OUTPUT,
        rate_limiter=rate_limiter,
    )
    exception_resolver.assert_not_called()
    adapter_factory.assert_called_once_with(
        model=primary_model,
        model_name="shared-model",
        fallback_model=None,
        fallback_exceptions=(),
    )


def test_evidence_agent_factory_uses_tool_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evidence agent should receive the tool-calling model profile."""

    settings = Settings(
        llm_model="default-model",
        llm_fast_model="fast-model",
        llm_quality_model="quality-model",
        llm_tool_model="tool-model",
    )
    repository = Mock()
    fake_model = Mock(spec=BaseChatModel)
    fake_agent = object()
    rate_limiter = Mock(spec=BaseRateLimiter)

    limiter_factory = Mock(return_value=rate_limiter)
    model_factory = Mock(return_value=fake_model)
    agent_factory = Mock(return_value=fake_agent)

    monkeypatch.setattr(
        factory,
        "_get_rate_limiter",
        limiter_factory,
    )
    monkeypatch.setattr(
        factory,
        "create_chat_model",
        model_factory,
    )
    monkeypatch.setattr(
        factory,
        "LangChainEvidenceDiscoveryAgent",
        agent_factory,
    )

    created_agent = factory.create_evidence_discovery_runner(
        repository=repository,
        settings=settings,
    )

    assert created_agent is fake_agent
    limiter_factory.assert_called_once_with(
        settings,
        ChatModelProfile.TOOL_CALLING,
    )
    model_factory.assert_called_once_with(
        settings=settings,
        profile=ChatModelProfile.TOOL_CALLING,
        rate_limiter=rate_limiter,
    )
    agent_factory.assert_called_once_with(
        repository=repository,
        settings=settings,
        model=fake_model,
        model_name="tool-model",
    )
