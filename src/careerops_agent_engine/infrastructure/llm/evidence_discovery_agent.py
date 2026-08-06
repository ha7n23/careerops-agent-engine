"""LangChain implementation of the CareerOps evidence agent."""

from collections.abc import Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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
    validate_evidence_discovery_result,
)
from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.llm.agent_trajectory import (
    inspect_evidence_agent_trajectory,
)


class LangChainEvidenceDiscoveryAgent:
    """Bounded tool-calling agent for approved career evidence."""

    def __init__(
        self,
        *,
        repository: EvidenceRepository,
        settings: Settings,
    ) -> None:
        """Create the model, tools and bounded agent harness."""

        model = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            thinking_level="low",
        )

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
                    exit_behavior="error",
                ),
                ToolCallLimitMiddleware(
                    tool_name="search_approved_evidence",
                    run_limit=(settings.evidence_agent_max_search_calls),
                    exit_behavior="error",
                ),
            ],
        )

        # The exact compiled-agent generic type is framework-internal and
        # version-sensitive, so Any is contained at this integration edge.
        self._agent: Any = create_agent(
            model=model,
            tools=tools,
            system_prompt=EVIDENCE_DISCOVERY_SYSTEM_PROMPT,
            response_format=EvidenceMatch,
            context_schema=EvidenceAgentContext,
            middleware=middleware,
            name="careerops_evidence_discovery_agent",
        )

        self._repository = repository
        self._recursion_limit = settings.evidence_agent_recursion_limit
        self._model_name = settings.llm_model

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Find and deterministically validate supporting evidence."""

        result: dict[str, Any] = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Analyse the following single job "
                            "requirement.\n\n"
                            "<job_requirement>\n"
                            f"{requirement.model_dump_json(indent=2)}\n"
                            "</job_requirement>"
                        ),
                    }
                ]
            },
            context=EvidenceAgentContext(user_id=user_id),
            config={
                "recursion_limit": self._recursion_limit,
                "run_name": "discover_requirement_evidence",
                "tags": [
                    "careerops",
                    "evidence-discovery",
                    "tool-calling-agent",
                ],
                "metadata": {
                    "requirement_id": requirement.requirement_id,
                    "prompt_version": PROMPT_VERSION,
                    "model_name": self._model_name,
                },
            },
        )

        match = EvidenceMatch.model_validate(result.get("structured_response"))

        messages = cast(
            list[BaseMessage],
            result.get("messages", []),
        )
        trajectory = inspect_evidence_agent_trajectory(messages)

        validate_evidence_discovery_result(
            requirement=requirement,
            match=match,
            called_tools=trajectory.called_tools,
            observed_evidence_ids=(trajectory.observed_evidence_ids),
            repository=self._repository,
            user_id=user_id,
        )

        return match
