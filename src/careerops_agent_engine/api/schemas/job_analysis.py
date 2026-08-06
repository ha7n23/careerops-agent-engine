"""API contracts for job-analysis operations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement


class JobAnalysisRequest(BaseModel):
    """Request to extract and analyse job requirements."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    job_id: str = Field(min_length=1, max_length=64)
    job_description: str = Field(
        min_length=1,
        max_length=50_000,
    )


class AuditEventResponse(BaseModel):
    """One visible graph execution event."""

    node: str
    event: str


class JobAnalysisResponse(BaseModel):
    """Successful result of an evidence-grounded job analysis."""

    status: Literal["completed"]
    job_id: str
    role_title: str | None

    requirements: list[JobRequirement]
    evidence_matches: list[EvidenceMatch]
    fit_score: float = Field(ge=0.0, le=100.0)

    audit_events: list[AuditEventResponse]
