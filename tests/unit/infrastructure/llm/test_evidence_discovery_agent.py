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
