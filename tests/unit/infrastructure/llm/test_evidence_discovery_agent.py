"""Unit tests for bounded batched evidence-discovery behaviour."""

from typing import Any
from unittest.mock import Mock

import pytest
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import (
    EvidenceMatch,
    EvidenceMatchBatch,
)
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


def build_no_match(requirement_id: str) -> EvidenceMatch:
    """Create one valid deterministic gap."""

    return EvidenceMatch(
        requirement_id=requirement_id,
        match_strength=MatchStrength.NONE,
        direct_evidence_ids=[],
        related_evidence_ids=[],
        explanation="No matching approved evidence exists.",
        gap=True,
    )


class _ModelLimitAgent:
    """Agent double that always exhausts its model-call budget."""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.call_count += 1

        raise ModelCallLimitExceededError(
            thread_count=0,
            run_count=6,
            thread_limit=None,
            run_limit=6,
        )


class _SuccessfulBatchAgent:
    """Agent double that returns one complete structured batch."""

    def __init__(self, batch: EvidenceMatchBatch) -> None:
        self._batch = batch
        self.call_count = 0
        self.last_input: dict[str, Any] | None = None

    def invoke(
        self,
        input_payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_input = input_payload

        return {
            "structured_response": self._batch,
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_approved_evidence",
                            "args": {
                                "query": "engineering requirements",
                                "limit": 10,
                            },
                            "id": "CALL-BATCH-001",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    name="search_approved_evidence",
                    tool_call_id="CALL-BATCH-001",
                    content=('{"query":"engineering requirements","records":[]}'),
                ),
            ],
        }


def test_agent_uses_batch_tool_strategy_for_structured_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool-calling agent must return one structured match batch."""

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
    assert response_format.schema is EvidenceMatchBatch


def test_model_call_limit_returns_fail_closed_gap() -> None:
    """A bounded single-requirement call must fail closed."""

    bounded_agent = _ModelLimitAgent()

    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._agent = bounded_agent
    agent._recursion_limit = 50
    agent._model_name = "test-model"

    result = agent.discover(
        build_requirement("REQ-PYTHON"),
        user_id="USER-TEST",
    )

    assert bounded_agent.call_count == 1
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


def test_non_empty_registry_uses_one_batch_agent_invocation() -> None:
    """All requirements should share one bounded agent run."""

    repository = Mock()
    repository.list_approved.return_value = [Mock()]

    batch_agent = _SuccessfulBatchAgent(
        EvidenceMatchBatch(
            matches=[
                build_no_match("REQ-002"),
                build_no_match("REQ-001"),
            ]
        )
    )

    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._repository = repository
    agent._agent = batch_agent
    agent._recursion_limit = 50
    agent._model_name = "test-model"

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

    assert batch_agent.call_count == 1
    assert batch_agent.last_input is not None

    prompt = batch_agent.last_input["messages"][0]["content"]

    assert "REQ-001" in prompt
    assert "REQ-002" in prompt

    assert [match.requirement_id for match in matches] == [
        "REQ-001",
        "REQ-002",
    ]


def test_batch_model_call_limit_returns_gap_for_every_requirement() -> None:
    """One exhausted batch run should fail closed for the whole set."""

    repository = Mock()
    repository.list_approved.return_value = [Mock()]

    bounded_agent = _ModelLimitAgent()

    agent = object.__new__(LangChainEvidenceDiscoveryAgent)
    agent._repository = repository
    agent._agent = bounded_agent
    agent._recursion_limit = 50
    agent._model_name = "test-model"

    matches = agent.discover_for_requirements(
        [
            build_requirement("REQ-001"),
            build_requirement("REQ-002"),
        ],
        user_id="USER-TEST",
    )

    assert bounded_agent.call_count == 1
    assert [match.requirement_id for match in matches] == [
        "REQ-001",
        "REQ-002",
    ]
    assert all(match.match_strength is MatchStrength.NONE for match in matches)
    assert all(match.gap for match in matches)
    assert all("bounded model-call limit" in match.explanation for match in matches)
