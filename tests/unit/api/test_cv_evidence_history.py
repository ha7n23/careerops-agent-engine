"""Tests for CV document and evidence-review history endpoints."""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_cv_evidence_history_service,
)
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
from careerops_agent_engine.main import app


class FakeDocumentHistoryRepository:
    """Return bounded document summaries for one user."""

    def __init__(
        self,
        summaries_by_user: dict[str, list[CareerDocumentSummary]],
    ) -> None:
        self._summaries_by_user = summaries_by_user

    def list_summaries(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[CareerDocumentSummary]:
        return list(self._summaries_by_user.get(user_id, []))[:limit]


class FakeReviewHistoryRepository:
    """Return bounded review summaries for one user."""

    def __init__(
        self,
        summaries_by_user: dict[str, list[CVEvidenceReviewRunSummary]],
    ) -> None:
        self._summaries_by_user = summaries_by_user

    def list_run_summaries(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[CVEvidenceReviewRunSummary]:
        return list(self._summaries_by_user.get(user_id, []))[:limit]


def build_document_summary(
    document_id: str,
    *,
    filename: str,
) -> CareerDocumentSummary:
    """Create one API document-history item."""

    timestamp = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)

    return CareerDocumentSummary(
        document_id=document_id,
        original_filename=filename,
        document_format=CareerDocumentFormat.PDF,
        size_bytes=125_000,
        status=CareerDocumentStatus.EXTRACTED,
        uploaded_at=timestamp,
        updated_at=timestamp,
    )


def build_review_summary(
    review_run_id: str,
    *,
    document_id: str,
) -> CVEvidenceReviewRunSummary:
    """Create one API review-history item."""

    timestamp = datetime(2026, 9, 24, 10, 5, tzinfo=UTC)

    return CVEvidenceReviewRunSummary(
        review_run_id=review_run_id,
        document_id=document_id,
        status=CVEvidenceReviewRunStatus.AWAITING_REVIEW,
        proposal_count=5,
        approved_evidence_count=0,
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a client with isolated history for two users."""

    document_repository = FakeDocumentHistoryRepository(
        {
            "USER-001": [
                build_document_summary(
                    "DOC-002",
                    filename="latest-cv.pdf",
                ),
                build_document_summary(
                    "DOC-001",
                    filename="master-cv.pdf",
                ),
            ],
            "USER-OTHER": [
                build_document_summary(
                    "DOC-PRIVATE",
                    filename="private-cv.pdf",
                )
            ],
        }
    )

    review_repository = FakeReviewHistoryRepository(
        {
            "USER-001": [
                build_review_summary(
                    "EVR-002",
                    document_id="DOC-002",
                ),
                build_review_summary(
                    "EVR-001",
                    document_id="DOC-001",
                ),
            ],
            "USER-OTHER": [
                build_review_summary(
                    "EVR-PRIVATE",
                    document_id="DOC-PRIVATE",
                )
            ],
        }
    )

    service = CVEvidenceHistoryService(
        document_repository=document_repository,
        review_repository=review_repository,
    )

    app.dependency_overrides[get_cv_evidence_history_service] = lambda: service

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(
            get_cv_evidence_history_service,
            None,
        )


def test_lists_frontend_safe_document_history(
    client: TestClient,
) -> None:
    """Document history should expose only the authenticated user's metadata."""

    response = client.get(
        "/api/v1/cv-documents",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert body["limit"] == 50

    assert [item["document_id"] for item in body["items"]] == [
        "DOC-002",
        "DOC-001",
    ]

    assert all("user_id" not in item for item in body["items"])
    assert all("storage_key" not in item for item in body["items"])
    assert all("sha256_hex" not in item for item in body["items"])


def test_lists_frontend_safe_review_history(
    client: TestClient,
) -> None:
    """Review history should expose status and useful counters."""

    response = client.get(
        "/api/v1/cv-evidence-reviews",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert body["items"][0]["review_run_id"] == "EVR-002"
    assert body["items"][0]["proposal_count"] == 5
    assert body["items"][0]["approved_evidence_count"] == 0
    assert "user_id" not in body["items"][0]


def test_history_endpoints_honour_requested_limit(
    client: TestClient,
) -> None:
    """Both public history endpoints should honour their bound."""

    documents = client.get(
        "/api/v1/cv-documents?limit=1",
        headers={"X-User-ID": "USER-001"},
    )

    reviews = client.get(
        "/api/v1/cv-evidence-reviews?limit=1",
        headers={"X-User-ID": "USER-001"},
    )

    assert documents.status_code == 200
    assert documents.json()["count"] == 1
    assert documents.json()["limit"] == 1

    assert reviews.status_code == 200
    assert reviews.json()["count"] == 1
    assert reviews.json()["limit"] == 1


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/cv-documents?limit=101",
        "/api/v1/cv-evidence-reviews?limit=101",
    ],
)
def test_history_endpoints_reject_unbounded_limits(
    client: TestClient,
    path: str,
) -> None:
    """Public history queries must reject excessive result sizes."""

    response = client.get(
        path,
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 422
