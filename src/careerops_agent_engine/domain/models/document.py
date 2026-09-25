"""Domain models for uploaded career documents."""

from datetime import datetime

from pydantic import Field

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVSection,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.base import DomainModel


class CareerDocument(DomainModel):
    """Metadata for one validated and securely stored document."""

    document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    original_filename: str = Field(
        min_length=1,
        max_length=255,
    )

    document_format: CareerDocumentFormat

    media_type: str = Field(
        min_length=1,
        max_length=128,
    )

    size_bytes: int = Field(
        ge=1,
    )

    sha256_hex: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    storage_key: str = Field(
        min_length=1,
        max_length=512,
    )

    status: CareerDocumentStatus


class CareerDocumentSummary(DomainModel):
    """Frontend-safe summary of one uploaded career document."""

    document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    original_filename: str = Field(
        min_length=1,
        max_length=255,
    )

    document_format: CareerDocumentFormat

    size_bytes: int = Field(
        ge=1,
    )

    status: CareerDocumentStatus

    uploaded_at: datetime

    updated_at: datetime


class ExtractedDocumentText(DomainModel):
    """Native text extracted from one validated career document."""

    document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    text: str = Field(
        min_length=1,
    )

    page_count: int | None = Field(
        default=None,
        ge=1,
    )

    paragraph_count: int | None = Field(
        default=None,
        ge=0,
    )

    warnings: list[str] = Field(default_factory=list)


class ParsedCVSection(DomainModel):
    """One deterministically identified section of an uploaded CV."""

    section: CVSection

    heading: str = Field(
        min_length=1,
        max_length=120,
    )

    text: str = Field(
        min_length=1,
    )

    order_index: int = Field(
        ge=0,
    )


class ParsedCVDocument(DomainModel):
    """Structured section view of extracted CV text."""

    document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    source_type: EvidenceSourceType = EvidenceSourceType.UPLOADED_CV

    preamble_text: str | None = None

    sections: list[ParsedCVSection] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
