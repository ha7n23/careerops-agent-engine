"""API schemas for career-document ingestion."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from careerops_agent_engine.application.services.cv_document_ingestion import (
    DEFAULT_MAX_TEXT_CHARACTERS,
    MAX_TEXT_TITLE_CHARACTERS,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    CareerDocumentSummary,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewRunSnapshot,
    CVEvidenceReviewRunSummary,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewResult,
)


class TextEvidenceSourceRequest(BaseModel):
    """Validated pasted-text evidence submitted by the frontend."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        max_length=MAX_TEXT_TITLE_CHARACTERS,
    )
    content: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_TEXT_CHARACTERS,
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


class CareerDocumentSummaryResponse(BaseModel):
    """Frontend-safe summary of one uploaded CV document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    original_filename: str
    document_format: CareerDocumentFormat
    size_bytes: int
    status: CareerDocumentStatus
    uploaded_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        summary: CareerDocumentSummary,
    ) -> "CareerDocumentSummaryResponse":
        """Build a public document-history item."""

        return cls(
            document_id=summary.document_id,
            original_filename=summary.original_filename,
            document_format=summary.document_format,
            size_bytes=summary.size_bytes,
            status=summary.status,
            uploaded_at=summary.uploaded_at,
            updated_at=summary.updated_at,
        )


class CareerDocumentHistoryResponse(BaseModel):
    """Bounded history of uploaded CV documents."""

    model_config = ConfigDict(extra="forbid")

    items: list[CareerDocumentSummaryResponse]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)

    @classmethod
    def from_domain(
        cls,
        summaries: list[CareerDocumentSummary],
        *,
        limit: int,
    ) -> "CareerDocumentHistoryResponse":
        """Build a bounded public document-history response."""

        items = [
            CareerDocumentSummaryResponse.from_domain(summary) for summary in summaries
        ]

        return cls(
            items=items,
            count=len(items),
            limit=limit,
        )


class CVEvidenceReviewRunSummaryResponse(BaseModel):
    """Frontend-safe summary of one evidence-review run."""

    model_config = ConfigDict(extra="forbid")

    review_run_id: str
    document_id: str
    status: CVEvidenceReviewRunStatus
    proposal_count: int
    approved_evidence_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        summary: CVEvidenceReviewRunSummary,
    ) -> "CVEvidenceReviewRunSummaryResponse":
        """Build a public review-history item."""

        return cls(
            review_run_id=summary.review_run_id,
            document_id=summary.document_id,
            status=summary.status,
            proposal_count=summary.proposal_count,
            approved_evidence_count=summary.approved_evidence_count,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class CVEvidenceReviewHistoryResponse(BaseModel):
    """Bounded history of CV evidence-review runs."""

    model_config = ConfigDict(extra="forbid")

    items: list[CVEvidenceReviewRunSummaryResponse]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)

    @classmethod
    def from_domain(
        cls,
        summaries: list[CVEvidenceReviewRunSummary],
        *,
        limit: int,
    ) -> "CVEvidenceReviewHistoryResponse":
        """Build a bounded public review-history response."""

        items = [
            CVEvidenceReviewRunSummaryResponse.from_domain(summary)
            for summary in summaries
        ]

        return cls(
            items=items,
            count=len(items),
            limit=limit,
        )
