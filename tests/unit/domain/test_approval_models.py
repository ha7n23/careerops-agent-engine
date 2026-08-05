"""Tests for human-review domain models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    ReviewAction,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
    CVReviewRecord,
    ProposalEdit,
)


def test_approval_requires_approved_proposals() -> None:
    """An empty approval cannot finalise a CV review."""

    with pytest.raises(
        ValidationError,
        match="requires approved proposals",
    ):
        CVReviewDecision(action=ReviewAction.APPROVE)


def test_edit_decision_requires_edited_text() -> None:
    """Selecting edit must include at least one concrete edit."""

    with pytest.raises(
        ValidationError,
        match="requires at least one edit",
    ):
        CVReviewDecision(action=ReviewAction.EDIT)


def test_proposal_cannot_be_approved_and_edited() -> None:
    """One proposal cannot receive conflicting review outcomes."""

    with pytest.raises(
        ValidationError,
        match="both approved and edited",
    ):
        CVReviewDecision(
            action=ReviewAction.EDIT,
            approved_proposal_ids=["CVP-001"],
            edits=[
                ProposalEdit(
                    proposal_id="CVP-001",
                    edited_text="Edited grounded proposal.",
                )
            ],
        )


def test_regeneration_requires_context() -> None:
    """Regeneration must identify rejected work or explain the request."""

    with pytest.raises(
        ValidationError,
        match="requires rejected proposals or a reviewer comment",
    ):
        CVReviewDecision(action=ReviewAction.REGENERATE)


def test_pending_review_cannot_have_completed_decision() -> None:
    """Pending reviews must not contain a hidden approval decision."""

    decision = CVReviewDecision(
        action=ReviewAction.APPROVE,
        approved_proposal_ids=["CVP-001"],
    )

    with pytest.raises(
        ValidationError,
        match="pending review cannot contain",
    ):
        CVReviewRecord(
            review_id="REV-001",
            application_pack_id="PACK-001",
            status=ApprovalStatus.PENDING,
            decision=decision,
            reviewed_at=datetime.now(UTC),
        )


def test_completed_review_status_must_match_action() -> None:
    """Stored review status must agree with the selected action."""

    decision = CVReviewDecision(
        action=ReviewAction.REJECT,
        rejected_proposal_ids=["CVP-001"],
    )

    with pytest.raises(
        ValidationError,
        match="status does not match",
    ):
        CVReviewRecord(
            review_id="REV-002",
            application_pack_id="PACK-001",
            status=ApprovalStatus.APPROVED,
            decision=decision,
            reviewed_at=datetime.now(UTC),
        )


def test_valid_edited_review_record() -> None:
    """An edited decision should produce a consistent review record."""

    decision = CVReviewDecision(
        action=ReviewAction.EDIT,
        approved_proposal_ids=["CVP-001"],
        edits=[
            ProposalEdit(
                proposal_id="CVP-002",
                edited_text=(
                    "Built a checkpointed LangGraph workflow with "
                    "human review controls."
                ),
            )
        ],
    )

    record = CVReviewRecord(
        review_id="REV-003",
        application_pack_id="PACK-001",
        status=ApprovalStatus.EDITED,
        decision=decision,
        reviewed_at=datetime.now(UTC),
    )

    assert record.status is ApprovalStatus.EDITED
    assert record.decision is not None
    assert record.decision.edits[0].proposal_id == "CVP-002"
