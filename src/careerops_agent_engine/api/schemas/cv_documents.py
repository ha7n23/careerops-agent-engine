"""API schemas for career-document ingestion."""

from pydantic import BaseModel, ConfigDict

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)


class CareerDocumentUploadResponse(BaseModel):
    """Safe metadata returned after a successful CV upload."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    original_filename: str
    document_format: CareerDocumentFormat
    media_type: str
    size_bytes: int
    sha256_hex: str
    status: CareerDocumentStatus

    @classmethod
    def from_domain(
        cls,
        document: CareerDocument,
    ) -> "CareerDocumentUploadResponse":
        """Build a public response without exposing storage internals."""

        return cls(
            document_id=document.document_id,
            original_filename=document.original_filename,
            document_format=document.document_format,
            media_type=document.media_type,
            size_bytes=document.size_bytes,
            sha256_hex=document.sha256_hex,
            status=document.status,
        )
