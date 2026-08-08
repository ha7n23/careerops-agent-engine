"""Domain models for uploaded career documents."""

from pydantic import Field

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
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
