"""Structured outputs returned by evidence-discovery tools."""

from typing import Self

from pydantic import Field

from careerops_agent_engine.domain.enums import EvidenceCategory
from careerops_agent_engine.domain.models.base import DomainModel
from careerops_agent_engine.domain.models.evidence import CareerEvidence


class EvidenceToolRecord(DomainModel):
    """Privacy-conscious evidence supplied to the discovery agent."""

    evidence_id: str
    category: EvidenceCategory
    title: str

    technologies: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    approved_claims: list[str] = Field(default_factory=list)

    @classmethod
    def from_evidence(
        cls,
        evidence: CareerEvidence,
    ) -> Self:
        """Create a bounded tool record from approved evidence."""

        return cls(
            evidence_id=evidence.evidence_id,
            category=evidence.category,
            title=evidence.title,
            technologies=evidence.technologies,
            capabilities=evidence.capabilities,
            approved_claims=evidence.approved_claims,
        )


class EvidenceSearchToolResult(DomainModel):
    """Bounded result of an approved-evidence search."""

    query: str
    records: list[EvidenceToolRecord]


class ProjectDetailsToolResult(DomainModel):
    """Result of retrieving one approved project record."""

    found: bool
    record: EvidenceToolRecord | None = None
    message: str


class VerifiedSkill(DomainModel):
    """Verified technology and the evidence supporting it."""

    name: str
    evidence_ids: list[str] = Field(min_length=1)


class VerifiedSkillsToolResult(DomainModel):
    """Technologies verified across approved evidence."""

    skills: list[VerifiedSkill]
