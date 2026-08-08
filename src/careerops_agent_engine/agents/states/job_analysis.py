"""State definitions for the job-analysis graph."""

from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict

JobAnalysisStatus = Literal[
    "validated",
    "invalid",
    "completed",
]


class AuditEvent(TypedDict):
    """One lightweight execution event accumulated in graph state."""

    node: str
    event: str


class JobAnalysisState(TypedDict):
    """Shared state for the job-analysis workflow."""

    job_id: str
    user_id: str
    job_description: str

    audit_events: Annotated[list[AuditEvent], add]

    role_title: NotRequired[str | None]

    requirements: NotRequired[list[dict[str, object]]]
    evidence_matches: NotRequired[list[dict[str, object]]]

    fit_score: NotRequired[float]

    cv_proposals: NotRequired[list[dict[str, object]]]
    claim_verification_reports: NotRequired[list[dict[str, object]]]

    reviewable_proposal_ids: NotRequired[list[str]]
    blocked_proposal_ids: NotRequired[list[str]]

    review_target_proposal_ids: NotRequired[list[str]]
    edited_proposal_ids: NotRequired[list[str]]
    edit_verification_failed_ids: NotRequired[list[str]]

    regenerated_proposal_ids: NotRequired[list[str]]
    regeneration_verification_failed_ids: NotRequired[list[str]]
    regeneration_feedback: NotRequired[str]

    review_decision: NotRequired[dict[str, object]]
    review_status: NotRequired[str]
    final_cv_proposals: NotRequired[list[dict[str, object]]]

    status: NotRequired[JobAnalysisStatus]
    validation_error: NotRequired[str | None]


class JobAnalysisUpdate(TypedDict, total=False):
    """Partial update returned by one graph node."""

    job_description: str
    role_title: str | None

    requirements: list[dict[str, object]]
    evidence_matches: list[dict[str, object]]

    fit_score: float

    cv_proposals: list[dict[str, object]]
    claim_verification_reports: list[dict[str, object]]

    reviewable_proposal_ids: list[str]
    blocked_proposal_ids: list[str]

    review_target_proposal_ids: list[str]
    edited_proposal_ids: list[str]
    edit_verification_failed_ids: list[str]

    regenerated_proposal_ids: list[str]
    regeneration_verification_failed_ids: list[str]
    regeneration_feedback: str

    review_decision: dict[str, object]
    review_status: str
    final_cv_proposals: list[dict[str, object]]

    status: JobAnalysisStatus
    validation_error: str | None

    audit_events: list[AuditEvent]
