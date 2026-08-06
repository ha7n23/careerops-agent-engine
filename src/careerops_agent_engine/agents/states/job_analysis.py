"""State definitions for the initial job-analysis graph."""

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
    """Shared state for the deterministic job-analysis workflow."""

    job_id: str
    job_description: str
    matched_requirement_ids: list[str]

    audit_events: Annotated[list[AuditEvent], add]

    requirements: NotRequired[list[dict[str, object]]]
    fit_score: NotRequired[float]
    status: NotRequired[JobAnalysisStatus]
    validation_error: NotRequired[str | None]


class JobAnalysisUpdate(TypedDict, total=False):
    """Partial update returned by one graph node."""

    job_description: str
    requirements: list[dict[str, object]]
    fit_score: float
    status: JobAnalysisStatus
    validation_error: str | None
    audit_events: list[AuditEvent]
