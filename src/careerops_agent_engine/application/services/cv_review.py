"""Deterministic validation for human CV review."""

from careerops_agent_engine.application.exceptions import (
    CVReviewValidationError,
)
from careerops_agent_engine.domain.enums import ReviewAction
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)


def validate_initial_review_decision(
    *,
    decision: CVReviewDecision,
    reviewable_proposal_ids: list[str],
) -> None:
    """Validate the first approve/reject review contract."""

    expected_ids = set(reviewable_proposal_ids)

    if not expected_ids:
        raise CVReviewValidationError(
            "Human review requires at least one reviewable proposal."
        )

    if decision.action is ReviewAction.APPROVE:
        if set(decision.approved_proposal_ids) != expected_ids:
            raise CVReviewValidationError(
                "An approval decision must approve every reviewable proposal."
            )

        if decision.rejected_proposal_ids or decision.edits:
            raise CVReviewValidationError(
                "An approval decision cannot reject or edit proposals."
            )

        return

    if decision.action is ReviewAction.REJECT:
        if set(decision.rejected_proposal_ids) != expected_ids:
            raise CVReviewValidationError(
                "A rejection decision must reject every reviewable proposal."
            )

        return

    raise CVReviewValidationError(
        "This workflow stage currently accepts only approve or reject decisions."
    )
