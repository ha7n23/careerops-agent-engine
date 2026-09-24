"""Deterministic validation for human CV review."""

from careerops_agent_engine.application.exceptions import (
    CVReviewValidationError,
)
from careerops_agent_engine.domain.enums import ReviewAction
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)


def validate_review_decision(
    *,
    decision: CVReviewDecision,
    reviewable_proposal_ids: list[str],
    allowed_actions: set[ReviewAction],
) -> None:
    """Validate a human decision against the current review target."""

    if decision.action not in allowed_actions:
        allowed_display = ", ".join(sorted(action.value for action in allowed_actions))

        raise CVReviewValidationError(
            f"This review stage accepts only: {allowed_display}."
        )

    expected_ids = set(reviewable_proposal_ids)

    if not expected_ids:
        raise CVReviewValidationError("Human review requires at least one proposal.")

    approved_ids = set(decision.approved_proposal_ids)
    rejected_ids = set(decision.rejected_proposal_ids)
    edited_ids = {edit.proposal_id for edit in decision.edits}

    referenced_ids = approved_ids | rejected_ids | edited_ids

    unknown_ids = referenced_ids - expected_ids

    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))

        raise CVReviewValidationError(
            "Review decision references proposals outside "
            "the current review set: "
            f"{unknown_display}"
        )

    if decision.action is ReviewAction.APPROVE:
        if edited_ids:
            raise CVReviewValidationError("An approval decision cannot edit proposals.")

        decided_ids = approved_ids | rejected_ids

        if decided_ids != expected_ids:
            raise CVReviewValidationError(
                "Every reviewable proposal must be approved or rejected."
            )

        return

    if decision.action is ReviewAction.REJECT:
        if rejected_ids != expected_ids:
            raise CVReviewValidationError(
                "A rejection decision must reject every reviewable proposal."
            )

        if approved_ids or edited_ids:
            raise CVReviewValidationError(
                "A rejection decision cannot approve or edit proposals."
            )

        return

    if decision.action is ReviewAction.EDIT:
        if approved_ids or rejected_ids:
            raise CVReviewValidationError(
                "This workflow stage does not yet support "
                "mixed approve, reject and edit decisions."
            )

        if edited_ids != expected_ids:
            raise CVReviewValidationError(
                "An edit decision must provide replacement "
                "text for every proposal under review."
            )

        return

    if decision.action is ReviewAction.REGENERATE:
        if approved_ids or edited_ids:
            raise CVReviewValidationError(
                "A regeneration decision cannot approve or edit proposals."
            )

        if rejected_ids != expected_ids:
            raise CVReviewValidationError(
                "A regeneration decision must target every proposal under review."
            )

        if not decision.reviewer_comment:
            raise CVReviewValidationError("Regeneration requires reviewer feedback.")

        return

    raise CVReviewValidationError("Unsupported human-review action.")
