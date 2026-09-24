"""Public API schemas for the approved Evidence Registry."""

from pydantic import BaseModel, ConfigDict, Field

from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)


class CareerEvidenceResponse(BaseModel):
    """Frontend-safe approved career evidence."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    category: EvidenceCategory
    title: str
    verification_status: VerificationStatus

    technologies: list[str]
    capabilities: list[str]
    approved_claims: list[str]
    source_references: list[SourceReference]

    @classmethod
    def from_domain(
        cls,
        evidence: CareerEvidence,
    ) -> "CareerEvidenceResponse":
        """Build a public representation of approved evidence."""

        return cls(
            evidence_id=evidence.evidence_id,
            category=evidence.category,
            title=evidence.title,
            verification_status=evidence.verification_status,
            technologies=list(evidence.technologies),
            capabilities=list(evidence.capabilities),
            approved_claims=list(evidence.approved_claims),
            source_references=list(evidence.source_references),
        )


class EvidenceRegistryListResponse(BaseModel):
    """Bounded list of approved evidence records."""

    model_config = ConfigDict(extra="forbid")

    items: list[CareerEvidenceResponse]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)

    @classmethod
    def from_domain(
        cls,
        evidence: list[CareerEvidence],
        *,
        limit: int,
    ) -> "EvidenceRegistryListResponse":
        """Build a bounded public registry response."""

        items = [CareerEvidenceResponse.from_domain(item) for item in evidence]

        return cls(
            items=items,
            count=len(items),
            limit=limit,
        )
