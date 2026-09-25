"""Tests for CV upload and evidence-review history queries."""

from datetime import UTC, datetime

import pytest

from careerops_agent_engine.application.services.cv_evidence_history import (
    CVEvidenceHistoryService,
)
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


class FakeDocumentHistoryRepository:
    """Record document-history queries and return fixed summaries."""

    def __init__(
        self,
        summaries: list[CareerDocumentSummary],
    ) -> None:
        self.summaries = summaries
        self.calls: list[tuple[str, int]] = []

    def list_summaries(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[CareerDocumentSummary]:
        self.calls.append((user_id, limit))
        return self.summaries[:limit]


class FakeReviewHistoryRepository:
    """Record review-history queries and return fixed summaries."""

    def __init__(
        self,
        summaries: list[CVEvidenceReviewRunSummary],
    ) -> None:
        self.summaries = summaries
        self.calls: list[tuple[str, int]] = []

    def list_run_summaries(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[CVEvidenceReviewRunSummary]:
        self.calls.append((user_id, limit))
        return self.summaries[:limit]


def build_service() -> tuple[
    CVEvidenceHistoryService,
    FakeDocumentHistoryRepository,
    FakeReviewHistoryRepository,
]:
    """Create one history service with deterministic repository data."""

    timestamp = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)

    document_repository = FakeDocumentHistoryRepository(
        [
            CareerDocumentSummary(
                document_id="DOC-001",
                original_filename="master-cv.pdf",
                document_format=CareerDocumentFormat.PDF,
                size_bytes=125_000,
                status=CareerDocumentStatus.EXTRACTED,
                uploaded_at=timestamp,
                updated_at=timestamp,
            )
        ]
    )

    review_repository = FakeReviewHistoryRepository(
        [
            CVEvidenceReviewRunSummary(
                review_run_id="EVR-001",
                document_id="DOC-001",
                status=CVEvidenceReviewRunStatus.AWAITING_REVIEW,
                proposal_count=5,
                approved_evidence_count=0,
                created_at=timestamp,
                updated_at=timestamp,
            )
        ]
    )

    return (
        CVEvidenceHistoryService(
            document_repository=document_repository,
            review_repository=review_repository,
        ),
        document_repository,
        review_repository,
    )


def test_lists_user_owned_document_history() -> None:
    """Document history should preserve the user and requested bound."""

    service, repository, _ = build_service()

    summaries = service.list_documents(
        user_id="USER-001",
        limit=25,
    )

    assert summaries[0].document_id == "DOC-001"
    assert repository.calls == [("USER-001", 25)]


def test_lists_user_owned_review_history() -> None:
    """Review history should preserve the user and requested bound."""

    service, _, repository = build_service()

    summaries = service.list_review_runs(
        user_id="USER-001",
        limit=25,
    )

    assert summaries[0].review_run_id == "EVR-001"
    assert repository.calls == [("USER-001", 25)]


@pytest.mark.parametrize("limit", [0, 101])
def test_rejects_invalid_history_limits(limit: int) -> None:
    """History queries must remain meaningfully bounded."""

    service, _, _ = build_service()

    with pytest.raises(
        ValueError,
        match="History limit must be between 1 and 100",
    ):
        service.list_documents(
            user_id="USER-001",
            limit=limit,
        )
