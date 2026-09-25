"""Tests for CV document and evidence-review history summaries."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocumentSummary,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewRunSummary,
)


def test_builds_career_document_summary() -> None:
    """A document summary should expose lightweight upload metadata."""

    uploaded_at = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    updated_at = datetime(2026, 9, 24, 10, 5, tzinfo=UTC)

    summary = CareerDocumentSummary(
        document_id="DOC-001",
        original_filename="master-cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        size_bytes=125_000,
        status=CareerDocumentStatus.EXTRACTED,
        uploaded_at=uploaded_at,
        updated_at=updated_at,
    )

    assert summary.document_id == "DOC-001"
    assert summary.original_filename == "master-cv.pdf"
    assert summary.uploaded_at == uploaded_at
    assert summary.updated_at == updated_at


def test_builds_cv_evidence_review_run_summary() -> None:
    """A review summary should expose status and useful counters."""

    created_at = datetime(2026, 9, 24, 10, 5, tzinfo=UTC)
    updated_at = datetime(2026, 9, 24, 10, 10, tzinfo=UTC)

    summary = CVEvidenceReviewRunSummary(
        review_run_id="EVR-001",
        document_id="DOC-001",
        status=CVEvidenceReviewRunStatus.COMPLETED,
        proposal_count=6,
        approved_evidence_count=4,
        created_at=created_at,
        updated_at=updated_at,
    )

    assert summary.review_run_id == "EVR-001"
    assert summary.proposal_count == 6
    assert summary.approved_evidence_count == 4


def test_history_summary_counts_cannot_be_negative() -> None:
    """Frontend counters must never contain impossible negative values."""

    timestamp = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)

    with pytest.raises(ValidationError):
        CVEvidenceReviewRunSummary(
            review_run_id="EVR-001",
            document_id="DOC-001",
            status=CVEvidenceReviewRunStatus.AWAITING_REVIEW,
            proposal_count=-1,
            approved_evidence_count=0,
            created_at=timestamp,
            updated_at=timestamp,
        )
