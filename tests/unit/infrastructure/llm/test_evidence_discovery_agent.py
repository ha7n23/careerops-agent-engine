"""Unit tests for bounded evidence-discovery behaviour."""

from typing import Any

from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
)

from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.job import JobRequirement
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
