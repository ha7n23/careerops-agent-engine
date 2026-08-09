"""Domain models for persistent CV evidence-review business history."""

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    CVEvidenceReviewRunStatus,
)
from careerops_agent_engine.domain.models.base import DomainModel
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
    EvidenceReviewResult,
)


class CVEvidenceReviewRunSnapshot(DomainModel):
    """Latest business snapshot of one CV evidence-review run."""

    review_run_id: str = Field(
        min_length=1,
        max_length=64,
    )

    user_id: str = Field(
        min_length=1,
        max_length=64,
    )

    document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    status: CVEvidenceReviewRunStatus

    proposals: list[CareerEvidenceProposal] = Field(default_factory=list)

    overlap_findings: list[CareerEvidenceOverlapFinding] = Field(default_factory=list)

    document_warnings: list[str] = Field(default_factory=list)

    review_result: EvidenceReviewResult | None = None

    @model_validator(mode="after")
    def validate_run_state(self) -> Self:
        """Ensure snapshot data agrees with the workflow status."""

        if (
            self.status is CVEvidenceReviewRunStatus.AWAITING_REVIEW
            and not self.proposals
        ):
            raise ValueError(
                "An awaiting-review CV evidence run requires pending proposals."
            )

        if self.status is CVEvidenceReviewRunStatus.COMPLETED and not self.proposals:
            raise ValueError("A completed CV evidence run requires source proposals.")

        if (
            self.status is CVEvidenceReviewRunStatus.COMPLETED
            and self.review_result is None
        ):
            raise ValueError(
                "A completed CV evidence run requires a human-review result."
            )

        if (
            self.status is not CVEvidenceReviewRunStatus.COMPLETED
            and self.review_result is not None
        ):
            raise ValueError(
                "Only a completed CV evidence run may contain a human-review result."
            )

        return self


class CVEvidenceReviewAuditEntry(DomainModel):
    """Append-only record of one applied evidence-review decision."""

    review_id: str = Field(
        min_length=1,
        max_length=64,
    )

    review_run_id: str = Field(
        min_length=1,
        max_length=64,
    )

    sequence_number: int = Field(
        ge=1,
    )

    decision: EvidenceReviewDecision

    result: EvidenceReviewResult

    recorded_at: datetime | None = None

    @model_validator(mode="after")
    def validate_decision_result_consistency(
        self,
    ) -> Self:
        """Ensure the audit result represents the recorded decision."""

        edited_ids = [edit.proposal_id for edit in self.decision.edits]

        if self.result.approved_proposal_ids != self.decision.approved_proposal_ids:
            raise ValueError(
                "Evidence audit approval result must match the human decision."
            )

        if self.result.rejected_proposal_ids != self.decision.rejected_proposal_ids:
            raise ValueError(
                "Evidence audit rejection result must match the human decision."
            )

        if self.result.edited_proposal_ids != edited_ids:
            raise ValueError(
                "Evidence audit edit result must match the human decision."
            )

        if (
            self.result.acknowledged_overlap_proposal_ids
            != self.decision.acknowledged_overlap_proposal_ids
        ):
            raise ValueError(
                "Evidence audit overlap acknowledgements must match the human decision."
            )

        return self
