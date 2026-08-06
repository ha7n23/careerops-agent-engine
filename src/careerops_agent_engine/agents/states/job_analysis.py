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
    status: NotRequired[JobAnalysisStatus]
    validation_error: NotRequired[str | None]


class JobAnalysisUpdate(TypedDict, total=False):
    """Partial update returned by one graph node."""

    job_description: str
    role_title: str | None
    requirements: list[dict[str, object]]
    evidence_matches: list[dict[str, object]]
    fit_score: float
    status: JobAnalysisStatus
    validation_error: str | None
    audit_events: list[AuditEvent]
