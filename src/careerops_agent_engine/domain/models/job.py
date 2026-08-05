"""Domain models for job-description analysis."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.base import DomainModel


class JobRequirement(DomainModel):
    """One requirement extracted from a job description."""

    requirement_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    category: RequirementCategory
    evidence_expected: str = Field(min_length=1, max_length=500)
    importance_score: int = Field(ge=1, le=5)
    source_text: str = Field(min_length=1, max_length=1_000)


class JobRequirementExtraction(DomainModel):
    """Validated result of job-requirement extraction."""

    role_title: str | None = Field(default=None, max_length=200)
    requirements: list[JobRequirement] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_requirement_ids(self) -> Self:
        """Reject duplicate requirement identifiers."""

        requirement_ids = [
            requirement.requirement_id for requirement in self.requirements
        ]

        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Requirement identifiers must be unique.")

        return self
