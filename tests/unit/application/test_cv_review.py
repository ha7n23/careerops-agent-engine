"""Tests for deterministic human CV-review validation."""

import pytest

from careerops_agent_engine.application.exceptions import (
    CVReviewValidationError,
)
from careerops_agent_engine.application.services.cv_review import (
    validate_review_decision,
)
from careerops_agent_engine.domain.enums import ReviewAction
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)


def test_accepts_partial_approval_with_rejected_remainder() -> None:
    """One decision may approve useful proposals and reject the rest."""

    decision = CVReviewDecision(
        action=ReviewAction.APPROVE,
        approved_proposal_ids=["CVP-001"],
        rejected_proposal_ids=["CVP-002"],
    )

    validate_review_decision(
        decision=decision,
        reviewable_proposal_ids=["CVP-001", "CVP-002"],
        allowed_actions={ReviewAction.APPROVE},
    )


def test_partial_approval_requires_complete_review_coverage() -> None:
    """Every reviewable proposal must still receive a decision."""

    decision = CVReviewDecision(
        action=ReviewAction.APPROVE,
        approved_proposal_ids=["CVP-001"],
    )

    with pytest.raises(
        CVReviewValidationError,
        match="Every reviewable proposal must be approved or rejected",
    ):
        validate_review_decision(
            decision=decision,
            reviewable_proposal_ids=["CVP-001", "CVP-002"],
            allowed_actions={ReviewAction.APPROVE},
        )


def test_accepts_approval_of_every_reviewable_proposal() -> None:
    """The existing approve-all path must remain valid."""

    decision = CVReviewDecision(
        action=ReviewAction.APPROVE,
        approved_proposal_ids=["CVP-001", "CVP-002"],
    )

    validate_review_decision(
        decision=decision,
        reviewable_proposal_ids=["CVP-001", "CVP-002"],
        allowed_actions={ReviewAction.APPROVE},
    )
