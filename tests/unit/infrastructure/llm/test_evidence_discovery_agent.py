"""Unit tests for bounded evidence-discovery behaviour."""

from typing import Any
from unittest.mock import Mock

import pytest
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.llm import (
    evidence_discovery_agent as evidence_discovery_agent_module,
)
from careerops_agent_engine.infrastructure.llm.evidence_discovery_agent import (
    LangChainEvidenceDiscoveryAgent,
)


def build_requirement(
    requirement_id: str,
) -> JobRequirement:
    """Create one evidence-discovery requirement."""

    return JobRequirement(
        requirement_id=requirement_id,
        name=f"Requirement {requirement_id}",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected="Approved engineering evidence.",
        importance_score=5,
        source_text="Engineering experience is required.",
    )


class _ModelLimitAgent:
    """Agent double that always exhausts its model-call budget."""

    def invoke(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        raise ModelCallLimitExceededError(
            thread_count=0,
            run_count=6,
            thread_limit=None,
            run_limit=6,
        )


def test_agent_uses_tool_strategy_for_structured_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-calling agents must not combine tools with provider JSON mode."""

    create_agent = Mock(return_value=Mock())

    monkeypatch.setattr(
        evidence_discovery_agent_module,
        "create_agent",
        create_agent,
    )

    LangChainEvidenceDiscoveryAgent(
        repository=Mock(),
        settings=Settings(),
        model=Mock(spec=BaseChatModel),
        model_name="tool-model",
    )

    response_format = create_agent.call_args.kwargs["response_format"]

    assert isinstance(response_format, ToolStrategy)
    assert response_format.schema is EvidenceMatch


def test_model_call_limit_returns_fail_closed_gap() -> None:
    """A bounded agent must not turn an exhausted budget into an API failure."""

    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._agent = _ModelLimitAgent()
    agent._recursion_limit = 50
    agent._model_name = "test-model"

    requirement = JobRequirement(
        requirement_id="REQ-PYTHON",
        name="Python",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected="Evidence of Python engineering.",
        importance_score=5,
        source_text="Strong Python programming skills are essential.",
    )

    result = agent.discover(
        requirement,
        user_id="USER-TEST",
    )

    assert result.requirement_id == "REQ-PYTHON"
    assert result.match_strength is MatchStrength.NONE
    assert result.direct_evidence_ids == []
    assert result.related_evidence_ids == []
    assert result.gap is True
    assert "bounded model-call limit" in result.explanation


def test_empty_requirement_set_skips_repository_and_agent() -> None:
    """No requirements should require no repository or model work."""

    repository = Mock()
    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._repository = repository
    agent._agent = Mock()

    matches = agent.discover_for_requirements(
        [],
        user_id="USER-TEST",
    )

    assert matches == []
    repository.list_approved.assert_not_called()
    agent._agent.invoke.assert_not_called()


def test_empty_registry_returns_gaps_without_agent_calls() -> None:
    """No approved evidence should deterministically skip the agent."""

    repository = Mock()
    repository.list_approved.return_value = []

    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._repository = repository
    agent._agent = Mock()

    matches = agent.discover_for_requirements(
        [
            build_requirement("REQ-001"),
            build_requirement("REQ-002"),
        ],
        user_id="USER-TEST",
    )

    repository.list_approved.assert_called_once_with(
        user_id="USER-TEST",
        limit=1,
    )
    agent._agent.invoke.assert_not_called()

    assert [match.requirement_id for match in matches] == [
        "REQ-001",
        "REQ-002",
    ]
    assert all(match.match_strength is MatchStrength.NONE for match in matches)
    assert all(match.gap for match in matches)
    assert all(match.direct_evidence_ids == [] for match in matches)


def test_non_empty_registry_preserves_per_requirement_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approved evidence should retain existing agent behaviour."""

    repository = Mock()
    repository.list_approved.return_value = [Mock()]

    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._repository = repository

    discover = Mock(
        side_effect=[
            EvidenceMatch(
                requirement_id="REQ-001",
                match_strength=MatchStrength.STRONG,
                direct_evidence_ids=["EVD-001"],
                related_evidence_ids=[],
                explanation="Direct evidence exists.",
                gap=False,
            ),
            EvidenceMatch(
                requirement_id="REQ-002",
                match_strength=MatchStrength.NONE,
                direct_evidence_ids=[],
                related_evidence_ids=[],
                explanation="No matching evidence exists.",
                gap=True,
            ),
        ]
    )

    monkeypatch.setattr(
        agent,
        "discover",
        discover,
    )

    matches = agent.discover_for_requirements(
        [
            build_requirement("REQ-001"),
            build_requirement("REQ-002"),
        ],
        user_id="USER-TEST",
    )

    repository.list_approved.assert_called_once_with(
        user_id="USER-TEST",
        limit=1,
    )
    assert discover.call_count == 2
    assert [match.requirement_id for match in matches] == [
        "REQ-001",
        "REQ-002",
    ]
