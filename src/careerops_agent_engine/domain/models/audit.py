"""Domain models for persistent job-analysis audit history."""

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    JobAnalysisRunStatus,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)
from careerops_agent_engine.domain.models.base import DomainModel
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
)


class JobAnalysisRunSnapshot(DomainModel):
    """Latest business snapshot of one durable job-analysis run."""

    thread_id: str = Field(
        min_length=1,
        max_length=64,
    )
    user_id: str = Field(
        min_length=1,
        max_length=64,
    )
    job_id: str = Field(
        min_length=1,
        max_length=64,
    )

    status: JobAnalysisRunStatus

    role_title: str | None = Field(
        default=None,
        max_length=250,
    )
    fit_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    review_status: ApprovalStatus | None = None

    cv_proposals: list[CVChangeProposal] = Field(default_factory=list)
    claim_verification_reports: list[CVClaimVerificationReport] = Field(
        default_factory=list
    )

    reviewable_proposal_ids: list[str] = Field(default_factory=list)
    blocked_proposal_ids: list[str] = Field(default_factory=list)

    final_cv_proposals: list[CVChangeProposal] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_consistency(
        self,
    ) -> Self:
        """Ensure final-review metadata appears only when meaningful."""

        if (
            self.status is not JobAnalysisRunStatus.COMPLETED
            and self.review_status is not None
        ):
            raise ValueError("Only a completed run may contain a final review status.")

        return self


class CVReviewAuditEntry(DomainModel):
    """Append-only business record of one applied human decision."""

    review_id: str = Field(
        min_length=1,
        max_length=64,
    )
    thread_id: str = Field(
        min_length=1,
        max_length=64,
    )

    sequence_number: int = Field(
        ge=1,
    )

    decision: CVReviewDecision

    result_status: JobAnalysisRunStatus
    result_review_status: ApprovalStatus | None = None

    resulting_cv_proposals: list[CVChangeProposal] = Field(default_factory=list)

    resulting_verification_reports: list[CVClaimVerificationReport] = Field(
        default_factory=list
    )

    recorded_at: datetime | None = None

    @model_validator(mode="after")
    def validate_result(
        self,
    ) -> Self:
        """Validate the business result of the human decision."""

        if self.result_status is JobAnalysisRunStatus.INVALID:
            raise ValueError("Human review cannot produce an invalid run.")

        if (
            self.result_status is JobAnalysisRunStatus.AWAITING_REVIEW
            and self.result_review_status is not None
        ):
            raise ValueError(
                "An awaiting-review result cannot have a final review status."
            )

        return self
