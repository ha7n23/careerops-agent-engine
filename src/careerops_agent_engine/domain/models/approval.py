"""Domain models for human review and approval."""

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    ReviewAction,
)
from careerops_agent_engine.domain.models.base import DomainModel


class ProposalEdit(DomainModel):
    """Human-edited replacement text for one CV proposal."""

    proposal_id: str = Field(min_length=1, max_length=64)
    edited_text: str = Field(min_length=1, max_length=1_500)


class CVReviewDecision(DomainModel):
    """Structured human decision for a set of CV proposals."""

    action: ReviewAction

    approved_proposal_ids: list[str] = Field(default_factory=list)
    rejected_proposal_ids: list[str] = Field(default_factory=list)
    edits: list[ProposalEdit] = Field(default_factory=list)

    reviewer_comment: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """Enforce clear and non-conflicting review semantics."""

        approved_ids = set(self.approved_proposal_ids)
        rejected_ids = set(self.rejected_proposal_ids)
        edited_ids = [edit.proposal_id for edit in self.edits]
        edited_id_set = set(edited_ids)

        if len(approved_ids) != len(self.approved_proposal_ids):
            raise ValueError("Approved proposal identifiers must be unique.")

        if len(rejected_ids) != len(self.rejected_proposal_ids):
            raise ValueError("Rejected proposal identifiers must be unique.")

        if len(edited_ids) != len(edited_id_set):
            raise ValueError("Edited proposal identifiers must be unique.")

        if approved_ids & rejected_ids:
            raise ValueError("A proposal cannot be both approved and rejected.")

        if approved_ids & edited_id_set:
            raise ValueError("A proposal cannot be both approved and edited.")

        if rejected_ids & edited_id_set:
            raise ValueError("A proposal cannot be both rejected and edited.")

        if self.action is ReviewAction.APPROVE:
            if not approved_ids:
                raise ValueError("An approval decision requires approved proposals.")
            if self.edits:
                raise ValueError("An approval decision cannot contain proposal edits.")

        if self.action is ReviewAction.EDIT and not self.edits:
            raise ValueError("An edit decision requires at least one edit.")

        if self.action is ReviewAction.REJECT:
            if not rejected_ids:
                raise ValueError("A rejection decision requires rejected proposals.")
            if approved_ids or self.edits:
                raise ValueError(
                    "A rejection decision cannot approve or edit proposals."
                )

        if self.action is ReviewAction.REGENERATE:
            if approved_ids or self.edits:
                raise ValueError(
                    "A regeneration request cannot approve or edit proposals."
                )
            if not rejected_ids and not self.reviewer_comment:
                raise ValueError(
                    "A regeneration request requires rejected proposals "
                    "or a reviewer comment."
                )

        return self


class CVReviewRecord(DomainModel):
    """Persistent metadata for one human-review request."""

    review_id: str = Field(min_length=1, max_length=64)
    application_pack_id: str = Field(min_length=1, max_length=64)
    status: ApprovalStatus

    decision: CVReviewDecision | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_status_and_decision(self) -> Self:
        """Ensure the stored status agrees with the human decision."""

        pending_statuses = {
            ApprovalStatus.NOT_REQUESTED,
            ApprovalStatus.PENDING,
        }

        if self.status in pending_statuses:
            if self.decision is not None or self.reviewed_at is not None:
                raise ValueError(
                    "A pending review cannot contain a completed decision."
                )
            return self

        expected_actions: dict[ApprovalStatus, ReviewAction] = {
            ApprovalStatus.APPROVED: ReviewAction.APPROVE,
            ApprovalStatus.EDITED: ReviewAction.EDIT,
            ApprovalStatus.REJECTED: ReviewAction.REJECT,
            ApprovalStatus.REGENERATION_REQUESTED: ReviewAction.REGENERATE,
        }

        if self.decision is None:
            raise ValueError("A completed review status requires a decision.")

        if self.reviewed_at is None:
            raise ValueError("A completed review status requires a review timestamp.")

        expected_action = expected_actions[self.status]

        if self.decision.action is not expected_action:
            raise ValueError("Review status does not match the recorded decision.")

        return self
