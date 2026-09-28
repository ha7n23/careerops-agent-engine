"""Public API schemas for the approved Evidence Registry."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRegistryPage,
)
from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceLifecycleStatus,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceEdit,
    SourceReference,
)

EvidenceEditValue = Annotated[str, Field(min_length=1, max_length=1_000)]


class CareerEvidenceEditRequest(BaseModel):
    """Strict frontend request for editable evidence fields only."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    category: EvidenceCategory | None = None
    title: str | None = Field(default=None, min_length=1, max_length=250)
    technologies: list[EvidenceEditValue] | None = Field(
        default=None,
        max_length=50,
    )
    capabilities: list[EvidenceEditValue] | None = Field(
        default=None,
        max_length=50,
    )
    approved_claims: list[EvidenceEditValue] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    @model_validator(mode="after")
    def require_actual_edit(self) -> Self:
        """Require at least one concrete replacement value."""

        if all(
            value is None
            for value in (
                self.category,
                self.title,
                self.technologies,
                self.capabilities,
                self.approved_claims,
            )
        ):
            raise ValueError("An evidence edit must change at least one field.")

        return self

    def to_domain(self) -> CareerEvidenceEdit:
        """Convert the public request to a strict domain edit."""

        return CareerEvidenceEdit.model_validate(self.model_dump(exclude_none=True))


class CareerEvidenceResponse(BaseModel):
    """Frontend-safe approved career evidence."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    category: EvidenceCategory
    title: str
    verification_status: VerificationStatus
    lifecycle_status: EvidenceLifecycleStatus

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
            lifecycle_status=evidence.lifecycle_status,
            technologies=list(evidence.technologies),
            capabilities=list(evidence.capabilities),
            approved_claims=list(evidence.approved_claims),
            source_references=list(evidence.source_references),
        )


class EvidenceRegistryListResponse(BaseModel):
    """Searchable, filterable page of approved evidence records."""

    model_config = ConfigDict(extra="forbid")

    items: list[CareerEvidenceResponse]
    count: int = Field(ge=0)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    has_more: bool

    @classmethod
    def from_domain(
        cls,
        page: EvidenceRegistryPage,
        *,
        limit: int,
        offset: int,
    ) -> "EvidenceRegistryListResponse":
        """Build public pagination metadata and safe evidence items."""

        items = [CareerEvidenceResponse.from_domain(item) for item in page.items]

        return cls(
            items=items,
            count=len(items),
            total=page.total,
            limit=limit,
            offset=offset,
            has_more=offset + len(items) < page.total,
        )
