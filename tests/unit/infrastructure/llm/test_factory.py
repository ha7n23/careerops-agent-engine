"""Tests for model-backed adapter factory wiring."""

from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter

from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.infrastructure.llm import factory
from careerops_agent_engine.infrastructure.llm.model_factory import (
    ChatModelProfile,
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
        (
            "create_cv_proposal_generator",
            "LangChainCVProposalGenerator",
        ),
        (
            "create_cv_claim_verifier",
            "LangChainCVClaimVerifier",
        ),
    ],
)
def test_structured_adapter_factory_uses_neutral_model(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    adapter_name: str,
) -> None:
    """Structured adapters should receive the configured neutral model."""

    settings = Settings(
        llm_provider="google",
        llm_model="gemini-test",
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
    limiter_factory.assert_called_once_with(settings)
    model_factory.assert_called_once_with(
        settings=settings,
        profile=ChatModelProfile.STRUCTURED_OUTPUT,
        rate_limiter=rate_limiter,
    )
    adapter_factory.assert_called_once_with(
        model=fake_model,
        model_name="gemini-test",
    )


def test_evidence_agent_factory_uses_tool_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evidence agent should receive the tool-calling model profile."""

    settings = Settings(
        llm_provider="google",
        llm_model="gemini-test",
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
    limiter_factory.assert_called_once_with(settings)
    model_factory.assert_called_once_with(
        settings=settings,
        profile=ChatModelProfile.TOOL_CALLING,
        rate_limiter=rate_limiter,
    )
    agent_factory.assert_called_once_with(
        repository=repository,
        settings=settings,
        model=fake_model,
        model_name="gemini-test",
    )
