"""Domain models for career evidence and requirement matching."""

from typing import Literal, Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.base import DomainModel


class SourceReference(DomainModel):
    """Traceable origin of a career evidence item."""

    source_type: EvidenceSourceType
    source_id: str = Field(min_length=1, max_length=128)
    page_number: int | None = Field(default=None, ge=1)
    source_excerpt: str | None = Field(default=None, max_length=1_000)


class CareerEvidenceCandidate(DomainModel):
    """Model-extracted evidence before trusted metadata is attached."""

    category: EvidenceCategory

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    source_section_order_index: int = Field(
        ge=0,
    )

    source_excerpt: str = Field(
        min_length=1,
        max_length=1_000,
    )

    technologies: list[str] = Field(default_factory=list)

    capabilities: list[str] = Field(default_factory=list)

    claims: list[str] = Field(min_length=1)

    warnings: list[str] = Field(default_factory=list)


class CareerEvidenceProposal(DomainModel):
    """Pending career evidence awaiting explicit human approval."""

    proposal_id: str = Field(
        min_length=1,
        max_length=64,
    )

    category: EvidenceCategory

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    verification_status: Literal[VerificationStatus.PENDING] = (
        VerificationStatus.PENDING
    )

    source_section: CVSection

    source_section_order_index: int = Field(
        ge=0,
    )

    technologies: list[str] = Field(default_factory=list)

    capabilities: list[str] = Field(default_factory=list)

    claims: list[str] = Field(min_length=1)

    source_references: list[SourceReference] = Field(min_length=1)

    warnings: list[str] = Field(default_factory=list)


class CareerEvidence(DomainModel):
    """Structured career information with verification and provenance."""

    evidence_id: str = Field(min_length=1, max_length=64)
    category: EvidenceCategory
    title: str = Field(min_length=1, max_length=250)
    verification_status: VerificationStatus

    technologies: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    approved_claims: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(min_length=1)


class EvidenceMatch(DomainModel):
    """Relationship between a requirement and available career evidence."""

    requirement_id: str = Field(min_length=1, max_length=64)
    match_strength: MatchStrength

    direct_evidence_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)

    explanation: str = Field(min_length=1, max_length=1_000)
    gap: bool

    @model_validator(mode="after")
    def validate_match_consistency(self) -> Self:
        """Enforce direct, related and missing-evidence semantics."""

        direct_ids = set(self.direct_evidence_ids)
        related_ids = set(self.related_evidence_ids)

        if direct_ids & related_ids:
            raise ValueError("The same evidence cannot be both direct and related.")

        if self.match_strength is MatchStrength.STRONG:
            if not direct_ids:
                raise ValueError("A strong match requires direct evidence.")
            if self.gap:
                raise ValueError("A strong match cannot be marked as a gap.")

        if self.match_strength is MatchStrength.PARTIAL:
            if not direct_ids:
                raise ValueError("A partial match requires direct evidence.")
            if not self.gap:
                raise ValueError("A partial match must identify a remaining gap.")

        if self.match_strength is MatchStrength.RELATED:
            if direct_ids:
                raise ValueError("A related match cannot contain direct evidence.")
            if not related_ids:
                raise ValueError("A related match requires related evidence.")
            if not self.gap:
                raise ValueError("A related match must remain a skill gap.")

        if self.match_strength is MatchStrength.NONE:
            if direct_ids or related_ids:
                raise ValueError("A no-match result cannot reference evidence.")
            if not self.gap:
                raise ValueError("A no-match result must be marked as a gap.")

        return self
