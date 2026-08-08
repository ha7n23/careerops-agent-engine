"""Application service for durable job-analysis workflows."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
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
from careerops_agent_engine.application.ports.job_analysis_audit_repository import (
    JobAnalysisAuditRepository,
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
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    JobAnalysisRunStatus,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)
from careerops_agent_engine.domain.models.audit import (
    CVReviewAuditEntry,
    JobAnalysisRunSnapshot,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
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
    """Run, resume, and persist the CareerOps job-analysis workflow."""

    def __init__(
        self,
        *,
        requirement_extractor: RequirementExtractor,
        evidence_discovery_runner: EvidenceDiscoveryRunner,
        cv_proposal_service: CVProposalGenerationService,
        cv_claim_verification_service: CVClaimVerificationService,
        checkpointer_factory: CheckpointerFactory,
        audit_repository: JobAnalysisAuditRepository,
    ) -> None:
        """Store workflow and business-persistence dependencies."""

        self._requirement_extractor = requirement_extractor
        self._evidence_discovery_runner = evidence_discovery_runner
        self._cv_proposal_service = cv_proposal_service
        self._cv_claim_verification_service = cv_claim_verification_service
        self._checkpointer_factory = checkpointer_factory
        self._audit_repository = audit_repository

    def analyse(
        self,
        *,
        job_id: str,
        user_id: str,
        job_description: str,
    ) -> JobAnalysisExecutionResult:
        """Start and persist one new durable job-analysis thread."""

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

        execution = build_execution_result(
            thread_id=thread_id,
            raw_result=raw_result,
        )

        self._audit_repository.save_run(build_run_snapshot(execution))

        return execution

    def resume_review(
        self,
        *,
        thread_id: str,
        user_id: str,
        decision: CVReviewDecision,
    ) -> JobAnalysisExecutionResult:
        """Resume and persist an authenticated human review."""

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

        execution = build_execution_result(
            thread_id=thread_id,
            raw_result=raw_result,
        )

        run_snapshot = build_run_snapshot(execution)

        existing_reviews = self._audit_repository.list_reviews(
            user_id=user_id,
            thread_id=thread_id,
        )

        sequence_number = next_review_sequence(existing_reviews)

        review = build_review_audit_entry(
            thread_id=thread_id,
            sequence_number=sequence_number,
            decision=decision,
            snapshot=run_snapshot,
        )

        self._audit_repository.save_review_result(
            snapshot=run_snapshot,
            review=review,
        )

        return execution

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


def build_review_id(
    *,
    thread_id: str,
    sequence_number: int,
) -> str:
    """Build a stable identifier for one review position."""

    digest = sha256(f"{thread_id}:{sequence_number}".encode()).hexdigest()[:16].upper()

    return f"REV-{digest}"


def next_review_sequence(
    reviews: list[CVReviewAuditEntry],
) -> int:
    """Return the next review sequence for one workflow thread."""

    if not reviews:
        return 1

    return max(review.sequence_number for review in reviews) + 1


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

        if not isinstance(
            interrupt_value,
            dict,
        ):
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


def build_run_snapshot(
    execution: JobAnalysisExecutionResult,
) -> JobAnalysisRunSnapshot:
    """Convert workflow execution into its latest business snapshot."""

    state = execution.state

    status = resolve_run_status(execution)

    review_status: ApprovalStatus | None = None

    if status is JobAnalysisRunStatus.COMPLETED:
        review_status_value = state.get("review_status")

        if review_status_value is not None:
            review_status = ApprovalStatus(review_status_value)

    proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get(
            "cv_proposals",
            [],
        )
    ]

    verification_reports = [
        CVClaimVerificationReport.model_validate(payload)
        for payload in state.get(
            "claim_verification_reports",
            [],
        )
    ]

    final_proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get(
            "final_cv_proposals",
            [],
        )
    ]

    return JobAnalysisRunSnapshot(
        thread_id=execution.thread_id,
        user_id=state["user_id"],
        job_id=state["job_id"],
        status=status,
        role_title=state.get("role_title"),
        fit_score=state.get("fit_score"),
        review_status=review_status,
        cv_proposals=proposals,
        claim_verification_reports=(verification_reports),
        reviewable_proposal_ids=list(
            state.get(
                "reviewable_proposal_ids",
                [],
            )
        ),
        blocked_proposal_ids=list(
            state.get(
                "blocked_proposal_ids",
                [],
            )
        ),
        final_cv_proposals=final_proposals,
    )


def resolve_run_status(
    execution: JobAnalysisExecutionResult,
) -> JobAnalysisRunStatus:
    """Map graph execution state to the business lifecycle."""

    if execution.awaiting_review:
        return JobAnalysisRunStatus.AWAITING_REVIEW

    state_status = execution.state.get("status")

    if state_status == "completed":
        return JobAnalysisRunStatus.COMPLETED

    if state_status == "invalid":
        return JobAnalysisRunStatus.INVALID

    raise RuntimeError(
        "CareerOps cannot persist a workflow that "
        "is neither paused, completed, nor invalid."
    )


def build_review_audit_entry(
    *,
    thread_id: str,
    sequence_number: int,
    decision: CVReviewDecision,
    snapshot: JobAnalysisRunSnapshot,
) -> CVReviewAuditEntry:
    """Build one append-only business review-history event."""

    return CVReviewAuditEntry(
        review_id=build_review_id(
            thread_id=thread_id,
            sequence_number=sequence_number,
        ),
        thread_id=thread_id,
        sequence_number=sequence_number,
        decision=decision,
        result_status=snapshot.status,
        result_review_status=(snapshot.review_status),
        resulting_cv_proposals=list(snapshot.cv_proposals),
        resulting_verification_reports=list(snapshot.claim_verification_reports),
    )
