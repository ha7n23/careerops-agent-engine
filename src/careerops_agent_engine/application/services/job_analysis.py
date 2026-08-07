"""Application service for durable job-analysis workflows."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from careerops_agent_engine.agents.graphs.job_analysis import (
    build_job_analysis_graph,
)
from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
)
from careerops_agent_engine.application.exceptions import (
    JobAnalysisThreadUnavailableError,
)
from careerops_agent_engine.application.ports.evidence_discovery import (
    EvidenceDiscoveryRunner,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)

CheckpointerFactory = Callable[
    [],
    AbstractContextManager[BaseCheckpointSaver[str]],
]


@dataclass(frozen=True)
class JobAnalysisExecutionResult:
    """Result of starting or resuming one durable graph thread."""

    thread_id: str
    state: JobAnalysisState
    interrupt_payload: dict[str, object] | None

    @property
    def awaiting_review(self) -> bool:
        """Whether execution is currently paused for review."""

        return self.interrupt_payload is not None


class JobAnalysisService:
    """Run and resume the CareerOps job-analysis workflow."""

    def __init__(
        self,
        *,
        requirement_extractor: RequirementExtractor,
        evidence_discovery_runner: EvidenceDiscoveryRunner,
        cv_proposal_service: CVProposalGenerationService,
        cv_claim_verification_service: (CVClaimVerificationService),
        checkpointer_factory: CheckpointerFactory,
    ) -> None:
        """Store workflow dependencies."""

        self._requirement_extractor = requirement_extractor
        self._evidence_discovery_runner = evidence_discovery_runner
        self._cv_proposal_service = cv_proposal_service
        self._cv_claim_verification_service = cv_claim_verification_service
        self._checkpointer_factory = checkpointer_factory

    def analyse(
        self,
        *,
        job_id: str,
        user_id: str,
        job_description: str,
    ) -> JobAnalysisExecutionResult:
        """Start one new durable job-analysis thread."""

        thread_id = build_thread_id()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        initial_state: JobAnalysisState = {
            "job_id": job_id,
            "user_id": user_id,
            "job_description": job_description,
            "audit_events": [],
        }

        with self._checkpointer_factory() as checkpointer:
            graph = self._build_graph(checkpointer)

            raw_result = cast(
                dict[str, Any],
                graph.invoke(
                    initial_state,
                    config=config,
                ),
            )

        return build_execution_result(
            thread_id=thread_id,
            raw_result=raw_result,
        )

    def resume_review(
        self,
        *,
        thread_id: str,
        user_id: str,
        decision: CVReviewDecision,
    ) -> JobAnalysisExecutionResult:
        """Resume a graph paused for authenticated human review."""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with self._checkpointer_factory() as checkpointer:
            graph = self._build_graph(checkpointer)

            snapshot = graph.get_state(config)

            snapshot_values = cast(
                dict[str, Any],
                snapshot.values,
            )

            if not snapshot_values:
                raise JobAnalysisThreadUnavailableError(
                    "The requested review thread is unavailable."
                )

            if snapshot_values.get("user_id") != user_id:
                raise JobAnalysisThreadUnavailableError(
                    "The requested review thread is unavailable."
                )

            allowed_review_nodes = {
                "request_human_review",
                "request_edit_rework",
            }

            if not (set(snapshot.next) & allowed_review_nodes):
                raise JobAnalysisThreadUnavailableError(
                    "The requested review thread is unavailable."
                )

            raw_result = cast(
                dict[str, Any],
                graph.invoke(
                    Command(resume=decision.model_dump(mode="json")),
                    config=config,
                ),
            )

        return build_execution_result(
            thread_id=thread_id,
            raw_result=raw_result,
        )

    def _build_graph(
        self,
        checkpointer: BaseCheckpointSaver[str],
    ) -> CompiledStateGraph[
        JobAnalysisState,
        None,
        JobAnalysisState,
        JobAnalysisState,
    ]:
        """Compile the graph against the active checkpointer."""

        return build_job_analysis_graph(
            requirement_extractor=(self._requirement_extractor),
            evidence_discovery_runner=(self._evidence_discovery_runner),
            cv_proposal_service=(self._cv_proposal_service),
            cv_claim_verification_service=(self._cv_claim_verification_service),
            checkpointer=checkpointer,
        )


def build_thread_id() -> str:
    """Create an opaque textual LangGraph thread identifier."""

    return f"THR-{uuid4().hex.upper()}"


def build_execution_result(
    *,
    thread_id: str,
    raw_result: dict[str, Any],
) -> JobAnalysisExecutionResult:
    """Separate graph state from an optional interrupt payload."""

    interrupt_payload: dict[str, object] | None = None

    interrupts = raw_result.get("__interrupt__")

    if interrupts:
        interrupt_value = interrupts[0].value

        if not isinstance(interrupt_value, dict):
            raise RuntimeError("CareerOps received an invalid interrupt payload.")

        interrupt_payload = cast(
            dict[str, object],
            interrupt_value,
        )

    state = cast(
        JobAnalysisState,
        {key: value for key, value in raw_result.items() if key != "__interrupt__"},
    )

    return JobAnalysisExecutionResult(
        thread_id=thread_id,
        state=state,
        interrupt_payload=interrupt_payload,
    )
