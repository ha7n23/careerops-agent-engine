"""Domain models for human review of extracted career evidence."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    VerificationStatus,
)
from careerops_agent_engine.domain.models.base import DomainModel
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)


class EvidenceProposalEdit(DomainModel):
    """Human corrections to one pending evidence proposal."""

    proposal_id: str = Field(
        min_length=1,
        max_length=64,
    )

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=250,
    )

    technologies: list[str] | None = None

    capabilities: list[str] | None = None

    claims: list[str] | None = Field(
        default=None,
        min_length=1,
    )

    @model_validator(mode="after")
    def require_actual_edit(self) -> Self:
        """Require at least one concrete replacement field."""

        if (
            self.title is None
            and self.technologies is None
            and self.capabilities is None
            and self.claims is None
        ):
            raise ValueError("An evidence edit must change at least one field.")

        return self


class EvidenceReviewDecision(DomainModel):
    """One human decision covering a batch of evidence proposals."""

    approved_proposal_ids: list[str] = Field(default_factory=list)

    rejected_proposal_ids: list[str] = Field(default_factory=list)

    edits: list[EvidenceProposalEdit] = Field(default_factory=list)

    acknowledged_overlap_proposal_ids: list[str] = Field(default_factory=list)

    reviewer_comment: str | None = Field(
        default=None,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def validate_decision_sets(self) -> Self:
        """Prevent duplicate and conflicting human decisions."""

        approved_ids = self.approved_proposal_ids
        rejected_ids = self.rejected_proposal_ids
        edited_ids = [edit.proposal_id for edit in self.edits]
        acknowledged_ids = self.acknowledged_overlap_proposal_ids

        if len(approved_ids) != len(set(approved_ids)):
            raise ValueError("Approved evidence proposal identifiers must be unique.")

        if len(rejected_ids) != len(set(rejected_ids)):
            raise ValueError("Rejected evidence proposal identifiers must be unique.")

        if len(edited_ids) != len(set(edited_ids)):
            raise ValueError("Edited evidence proposal identifiers must be unique.")

        if len(acknowledged_ids) != len(set(acknowledged_ids)):
            raise ValueError("Acknowledged overlap identifiers must be unique.")

        approved_set = set(approved_ids)
        rejected_set = set(rejected_ids)
        edited_set = set(edited_ids)

        if approved_set & rejected_set:
            raise ValueError(
                "An evidence proposal cannot be both approved and rejected."
            )

        if approved_set & edited_set:
            raise ValueError("An evidence proposal cannot be both approved and edited.")

        if rejected_set & edited_set:
            raise ValueError("An evidence proposal cannot be both rejected and edited.")

        if not (approved_set or rejected_set or edited_set):
            raise ValueError("Evidence review requires at least one proposal decision.")

        return self


class EvidenceReviewResult(DomainModel):
    """Deterministic result of completed human evidence review."""

    approved_proposal_ids: list[str] = Field(default_factory=list)

    edited_proposal_ids: list[str] = Field(default_factory=list)

    rejected_proposal_ids: list[str] = Field(default_factory=list)

    acknowledged_overlap_proposal_ids: list[str] = Field(default_factory=list)

    approved_evidence: list[CareerEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Ensure only approved evidence appears in the result."""

        accepted_ids = {
            *self.approved_proposal_ids,
            *self.edited_proposal_ids,
        }

        if len(self.approved_evidence) != len(accepted_ids):
            raise ValueError(
                "Every accepted proposal must produce "
                "exactly one approved evidence record."
            )

        if any(
            evidence.verification_status is not VerificationStatus.APPROVED
            for evidence in self.approved_evidence
        ):
            raise ValueError(
                "Evidence review results may expose only approved evidence records."
            )

        evidence_ids = [evidence.evidence_id for evidence in self.approved_evidence]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Approved evidence identifiers must be unique.")

        return self
