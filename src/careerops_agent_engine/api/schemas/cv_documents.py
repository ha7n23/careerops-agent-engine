"""API schemas for career-document ingestion."""

from pydantic import BaseModel, ConfigDict

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewResult,
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


class CVEvidenceReviewRunResponse(BaseModel):
    """Public state of one durable CV evidence-review run."""

    model_config = ConfigDict(extra="forbid")

    review_run_id: str
    document_id: str
    status: CVEvidenceReviewRunStatus

    proposals: list[CareerEvidenceProposal]
    overlap_findings: list[CareerEvidenceOverlapFinding]
    document_warnings: list[str]

    review_result: EvidenceReviewResult | None = None

    @classmethod
    def from_domain(
        cls,
        snapshot: CVEvidenceReviewRunSnapshot,
    ) -> "CVEvidenceReviewRunResponse":
        """Build a safe response without exposing the owning user ID."""

        return cls(
            review_run_id=snapshot.review_run_id,
            document_id=snapshot.document_id,
            status=snapshot.status,
            proposals=list(snapshot.proposals),
            overlap_findings=list(snapshot.overlap_findings),
            document_warnings=list(snapshot.document_warnings),
            review_result=snapshot.review_result,
        )
