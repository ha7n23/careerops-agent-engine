"""API contracts for durable job-analysis operations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    ReviewAction,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
)


class JobAnalysisRequest(BaseModel):
    """Request to start one job-analysis workflow."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    job_id: str = Field(
        min_length=1,
        max_length=64,
    )
    job_description: str = Field(
        min_length=1,
        max_length=50_000,
    )


class AuditEventResponse(BaseModel):
    """One visible graph execution event."""

    node: str
    event: str


class CVProposalReviewResponse(BaseModel):
    """Human-review payload exposed when the graph pauses."""

    type: Literal["cv_proposal_review"]

    proposals: list[CVChangeProposal]
    verification_reports: list[CVClaimVerificationReport]

    allowed_actions: list[ReviewAction]


class JobAnalysisResultBase(BaseModel):
    """Fields shared by paused and completed analyses."""

    thread_id: str

    job_id: str
    role_title: str | None

    requirements: list[JobRequirement]
    evidence_matches: list[EvidenceMatch]

    fit_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    cv_proposals: list[CVChangeProposal]
    claim_verification_reports: list[CVClaimVerificationReport]

    reviewable_proposal_ids: list[str]
    blocked_proposal_ids: list[str]

    audit_events: list[AuditEventResponse]


class JobAnalysisAwaitingReviewResponse(JobAnalysisResultBase):
    """Response returned when execution pauses for review."""

    status: Literal["awaiting_review"]
    review: CVProposalReviewResponse


class JobAnalysisCompletedResponse(JobAnalysisResultBase):
    """Response returned after the graph reaches completion."""

    status: Literal["completed"]

    review_status: ApprovalStatus | None = None

    final_cv_proposals: list[CVChangeProposal] = Field(default_factory=list)


JobAnalysisResponse = JobAnalysisAwaitingReviewResponse | JobAnalysisCompletedResponse
