"""Provider-facing schemas for structured LLM operations."""

from pydantic import Field

from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.base import DomainModel


class ExtractedRequirement(DomainModel):
    """One model-extracted requirement before IDs are assigned."""

    name: str = Field(min_length=1, max_length=200)
    category: RequirementCategory
    evidence_expected: str = Field(
        min_length=1,
        max_length=500,
    )
    importance_score: int = Field(ge=1, le=5)
    source_text: str = Field(
        min_length=1,
        max_length=1_000,
    )


class ExtractedRequirementSet(DomainModel):
    """Raw structured result returned by the model provider."""

    role_title: str | None = Field(
        default=None,
        max_length=200,
    )
    requirements: list[ExtractedRequirement] = Field(min_length=1)


class ProposalEvidenceContext(DomainModel):
    """Approved evidence exposed to the CV proposal model."""

    evidence_id: str = Field(min_length=1, max_length=64)
    category: EvidenceCategory
    title: str = Field(min_length=1, max_length=250)

    technologies: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    approved_claims: list[str] = Field(default_factory=list)


class GeneratedCVProposalContent(DomainModel):
    """Model-generated content before domain metadata is attached."""

    section: CVSection

    proposed_text: str = Field(
        min_length=1,
        max_length=1_500,
    )

    supporting_evidence_ids: list[str] = Field(min_length=1)

    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    warnings: list[str] = Field(default_factory=list)


class GeneratedClaimAssessment(DomainModel):
    """Provider assessment of one factual claim."""

    claim_text: str = Field(
        min_length=1,
        max_length=1_500,
    )
    supported: bool

    supporting_evidence_ids: list[str] = Field(default_factory=list)

    explanation: str = Field(
        min_length=1,
        max_length=1_500,
    )


class GeneratedClaimVerification(DomainModel):
    """Provider result covering all factual proposal claims."""

    claims: list[GeneratedClaimAssessment] = Field(min_length=1)

    coverage_complete: bool

    coverage_notes: list[str] = Field(default_factory=list)


class ExtractedCareerEvidenceCandidate(DomainModel):
    """One provider-extracted CV evidence candidate."""

    category: EvidenceCategory
    title: str
    source_section_order_index: int
    source_excerpt: str

    technologies: list[str]
    capabilities: list[str]
    claims: list[str]
    warnings: list[str]


class ExtractedCareerEvidenceSet(DomainModel):
    """Structured evidence candidates returned by the provider."""

    candidates: list[ExtractedCareerEvidenceCandidate]
