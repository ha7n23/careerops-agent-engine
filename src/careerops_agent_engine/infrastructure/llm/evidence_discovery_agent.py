"""LangChain implementation of the CareerOps evidence agent."""

from collections.abc import Sequence
from json import dumps
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from careerops_agent_engine.agents.prompts.evidence_discovery import (
    EVIDENCE_DISCOVERY_SYSTEM_PROMPT,
    PROMPT_VERSION,
)
from careerops_agent_engine.agents.runtime_context import (
    EvidenceAgentContext,
)
from careerops_agent_engine.agents.tools.evidence import (
    create_evidence_tools,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.services.evidence_validation import (
    validate_evidence_discovery_batch,
)
from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.domain.enums import MatchStrength
from careerops_agent_engine.domain.models.evidence import (
    EvidenceMatch,
    EvidenceMatchBatch,
)
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.llm.agent_trajectory import (
    inspect_evidence_agent_trajectory,
)
from careerops_agent_engine.infrastructure.observability.langsmith import (
    build_langsmith_run_config,
)


def build_empty_registry_match(
    requirement: JobRequirement,
) -> EvidenceMatch:
    """Create a deterministic gap when no approved evidence exists."""

    return EvidenceMatch(
        requirement_id=requirement.requirement_id,
        match_strength=MatchStrength.NONE,
        direct_evidence_ids=[],
        related_evidence_ids=[],
        explanation=(
            "No human-approved career evidence is available for evidence discovery."
        ),
        gap=True,
    )


def build_model_limit_match(
    requirement: JobRequirement,
) -> EvidenceMatch:
    """Create a fail-closed gap when the batch model budget is exhausted."""

    return EvidenceMatch(
        requirement_id=requirement.requirement_id,
        match_strength=MatchStrength.NONE,
        direct_evidence_ids=[],
        related_evidence_ids=[],
        explanation=(
            "Evidence discovery reached its bounded model-call limit "
            "before a supported match could be completed."
        ),
        gap=True,
    )


def build_requirement_batch_prompt(
    requirements: Sequence[JobRequirement],
) -> str:
    """Serialize one untrusted requirement batch inside clear delimiters."""

    payload = [requirement.model_dump(mode="json") for requirement in requirements]

    return (
        "Analyse the following job requirement batch and return "
        "exactly one evidence match for every requirement.\n\n"
        "<job_requirements>\n"
        f"{dumps(payload, indent=2)}\n"
        "</job_requirements>"
    )


class LangChainEvidenceDiscoveryAgent:
    """Bounded tool-calling agent for approved career evidence."""

    def __init__(
        self,
        *,
        repository: EvidenceRepository,
        settings: Settings,
        model: BaseChatModel,
        model_name: str,
    ) -> None:
        """Create the tools and bounded batch-agent harness."""

        tools = create_evidence_tools(repository)

        # LangChain middleware classes use different internal state and
        # context generic parameters. The framework supports combining
        # them, but static type checkers cannot infer one common invariant
        # middleware type. Keep the cast at this framework boundary.
        middleware = cast(
            Sequence[
                AgentMiddleware[
                    Any,
                    EvidenceAgentContext,
                    Any,
                ]
            ],
            [
                ModelCallLimitMiddleware(
                    run_limit=(settings.evidence_agent_max_model_calls),
                    exit_behavior="error",
                ),
                ToolCallLimitMiddleware(
                    run_limit=(settings.evidence_agent_max_tool_calls),
                    exit_behavior="continue",
                ),
                ToolCallLimitMiddleware(
                    tool_name="search_approved_evidence",
                    run_limit=(settings.evidence_agent_max_search_calls),
                    exit_behavior="continue",
                ),
            ],
        )

        # The exact compiled-agent generic type is framework-internal and
        # version-sensitive, so Any is contained at this integration edge.
        self._agent: Any = create_agent(
            model=model,
            tools=tools,
            system_prompt=EVIDENCE_DISCOVERY_SYSTEM_PROMPT,
            response_format=ToolStrategy(
                schema=EvidenceMatchBatch,
            ),
            context_schema=EvidenceAgentContext,
            middleware=middleware,
            name="careerops_evidence_discovery_agent",
        )

        self._repository = repository
        self._recursion_limit = settings.evidence_agent_recursion_limit
        self._model_name = model_name

    def discover_for_requirements(
        self,
        requirements: Sequence[JobRequirement],
        *,
        user_id: str,
    ) -> list[EvidenceMatch]:
        """Discover all matches through at most one bounded agent run."""

        requirement_list = list(requirements)

        if not requirement_list:
            return []

        approved_evidence_exists = bool(
            self._repository.list_approved(
                user_id=user_id,
                limit=1,
            )
        )

        if not approved_evidence_exists:
            return [
                build_empty_registry_match(requirement)
                for requirement in requirement_list
            ]

        return self._discover_batch(
            requirement_list,
            user_id=user_id,
        )

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Discover one match through the shared batch implementation."""

        return self._discover_batch(
            [requirement],
            user_id=user_id,
        )[0]

    def _discover_batch(
        self,
        requirements: Sequence[JobRequirement],
        *,
        user_id: str,
    ) -> list[EvidenceMatch]:
        """Run and deterministically validate one requirement batch."""

        requirement_list = list(requirements)

        if not requirement_list:
            return []

        try:
            result: dict[str, Any] = self._agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": build_requirement_batch_prompt(requirement_list),
                        }
                    ]
                },
                context=EvidenceAgentContext(user_id=user_id),
                config={
                    **build_langsmith_run_config(
                        run_name=("discover_requirement_evidence_batch"),
                        tags=[
                            "evidence-discovery",
                            "batch",
                            "tool-calling-agent",
                            "llm",
                        ],
                        metadata={
                            "component": ("evidence_discovery_agent"),
                            "requirement_count": len(requirement_list),
                            "prompt_version": PROMPT_VERSION,
                            "ls_model_name": self._model_name,
                        },
                    ),
                    "recursion_limit": self._recursion_limit,
                },
            )
        except ModelCallLimitExceededError:
            return [
                build_model_limit_match(requirement) for requirement in requirement_list
            ]

        batch = EvidenceMatchBatch.model_validate(result.get("structured_response"))

        messages = cast(
            list[BaseMessage],
            result.get("messages", []),
        )

        trajectory = inspect_evidence_agent_trajectory(messages)

        return validate_evidence_discovery_batch(
            requirements=requirement_list,
            matches=batch.matches,
            called_tools=trajectory.called_tools,
            observed_evidence_ids=(trajectory.observed_evidence_ids),
            repository=self._repository,
            user_id=user_id,
        )
