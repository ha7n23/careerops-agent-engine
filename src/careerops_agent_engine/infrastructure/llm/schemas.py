"""Provider-facing schemas for structured LLM extraction."""

from pydantic import Field

from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.base import DomainModel


class ExtractedRequirement(DomainModel):
    """One model-extracted requirement before IDs are assigned."""

    name: str = Field(min_length=1, max_length=200)
    category: RequirementCategory
    evidence_expected: str = Field(min_length=1, max_length=500)
    importance_score: int = Field(ge=1, le=5)
    source_text: str = Field(min_length=1, max_length=1_000)


class ExtractedRequirementSet(DomainModel):
    """Raw structured result returned by the model provider."""

    role_title: str | None = Field(default=None, max_length=200)
    requirements: list[ExtractedRequirement] = Field(min_length=1)
