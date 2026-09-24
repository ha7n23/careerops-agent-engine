"""Domain models for career evidence and requirement matching."""

from typing import Literal, Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceOverlapScope,
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


class CareerEvidenceOverlapFinding(DomainModel):
    """Deterministic potential duplication requiring review."""

    proposal_id: str = Field(
        min_length=1,
        max_length=64,
    )

    scope: EvidenceOverlapScope

    matching_proposal_id: str | None = Field(
        default=None,
        max_length=64,
    )

    matching_evidence_id: str | None = Field(
        default=None,
        max_length=64,
    )

    matched_claims: list[str] = Field(default_factory=list)

    same_source_excerpt: bool = False

    @model_validator(mode="after")
    def validate_overlap_target(self) -> Self:
        """Require the correct comparison identifier for each scope."""

        if self.scope is EvidenceOverlapScope.WITHIN_DOCUMENT:
            if self.matching_proposal_id is None:
                raise ValueError(
                    "Within-document overlap requires a matching proposal identifier."
                )

            if self.matching_evidence_id is not None:
                raise ValueError(
                    "Within-document overlap cannot reference approved evidence."
                )

            if self.matching_proposal_id == self.proposal_id:
                raise ValueError("A proposal cannot overlap with itself.")

        if self.scope is EvidenceOverlapScope.APPROVED_EVIDENCE:
            if self.matching_evidence_id is None:
                raise ValueError(
                    "Approved-evidence overlap requires a matching evidence identifier."
                )

            if self.matching_proposal_id is not None:
                raise ValueError(
                    "Approved-evidence overlap cannot "
                    "reference another pending proposal."
                )

        if not self.matched_claims and not self.same_source_excerpt:
            raise ValueError(
                "Evidence overlap requires a deterministic matching signal."
            )

        return self


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


class EvidenceMatchBatch(DomainModel):
    """Structured evidence matches returned for one requirement batch."""

    matches: list[EvidenceMatch] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_requirement_ids(self) -> Self:
        """Require at most one match for each requirement."""

        requirement_ids = [match.requirement_id for match in self.matches]

        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError(
                "Batch evidence matches must have unique requirement identifiers."
            )

        return self
