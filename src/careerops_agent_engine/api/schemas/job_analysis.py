"""API contracts for job-analysis operations."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from careerops_agent_engine.domain.models.job import JobRequirement


class JobAnalysisRequest(BaseModel):
    """Request to extract and analyse job requirements."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    job_id: str = Field(min_length=1, max_length=64)
    job_description: str = Field(min_length=1, max_length=50_000)

    matched_requirement_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Temporary development input representing requirements "
            "already supported by direct evidence."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_match_ids(self) -> Self:
        """Reject duplicated requirement references."""

        if len(self.matched_requirement_ids) != len(set(self.matched_requirement_ids)):
            raise ValueError("Matched requirement identifiers must be unique.")

        return self


class AuditEventResponse(BaseModel):
    """One visible graph execution event."""

    node: str
    event: str


class JobAnalysisResponse(BaseModel):
    """Successful result of a job-analysis workflow."""

    status: Literal["completed"]
    job_id: str
    role_title: str | None

    requirements: list[JobRequirement]
    fit_score: float = Field(ge=0.0, le=100.0)

    audit_events: list[AuditEventResponse]
