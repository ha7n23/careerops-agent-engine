"""Tests for the durable CV evidence-review API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_cv_evidence_workflow_service,
)
from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    CVEvidenceReviewRunUnavailableError,
)
from careerops_agent_engine.application.services.cv_evidence_review import (
    CVEvidenceReviewService,
)
from careerops_agent_engine.domain.enums import (
    CVEvidenceReviewRunStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceProposal,
    SourceReference,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
)
from careerops_agent_engine.main import app


class FakeCVEvidenceWorkflowService:
    """Keep one durable evidence-review workflow in memory."""

    def __init__(self) -> None:
        """Create one awaiting-review snapshot."""

        self.awaiting_snapshot = CVEvidenceReviewRunSnapshot(
            review_run_id="EVR-API-001",
            user_id="USER-001",
            document_id="DOC-001",
            status=(CVEvidenceReviewRunStatus.AWAITING_REVIEW),
            proposals=[build_proposal()],
            overlap_findings=[],
            document_warnings=[],
        )

        self.completed_snapshot: CVEvidenceReviewRunSnapshot | None = None

        self.last_decision: EvidenceReviewDecision | None = None

        self.review_write_count = 0

    def start_review(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CVEvidenceReviewRunSnapshot:
        """Start or recover the user-owned review."""

        if user_id != "USER-001" or document_id != "DOC-001":
            raise CareerDocumentUnavailableError("The career document is unavailable.")

        if self.completed_snapshot is not None:
            return self.completed_snapshot

        return self.awaiting_snapshot

    def get_review(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> CVEvidenceReviewRunSnapshot:
        """Recover one user-owned review run."""

        self._validate_review_access(
            user_id=user_id,
            review_run_id=review_run_id,
        )

        if self.completed_snapshot is not None:
            return self.completed_snapshot

        return self.awaiting_snapshot

    def submit_review(
        self,
        *,
        user_id: str,
        review_run_id: str,
        decision: EvidenceReviewDecision,
    ) -> CVEvidenceReviewRunSnapshot:
        """Apply one human decision with idempotent retry behavior."""

        self._validate_review_access(
            user_id=user_id,
            review_run_id=review_run_id,
        )

        if self.completed_snapshot is not None:
            if decision == self.last_decision:
                return self.completed_snapshot

            raise CVEvidenceReviewRunUnavailableError(
                "The CV evidence review run is unavailable."
            )

        result = CVEvidenceReviewService().review(
            proposals=list(self.awaiting_snapshot.proposals),
            overlap_findings=list(self.awaiting_snapshot.overlap_findings),
            decision=decision,
        )

        self.completed_snapshot = CVEvidenceReviewRunSnapshot(
            review_run_id=(self.awaiting_snapshot.review_run_id),
            user_id=(self.awaiting_snapshot.user_id),
            document_id=(self.awaiting_snapshot.document_id),
            status=(CVEvidenceReviewRunStatus.COMPLETED),
            proposals=list(self.awaiting_snapshot.proposals),
            overlap_findings=list(self.awaiting_snapshot.overlap_findings),
            document_warnings=list(self.awaiting_snapshot.document_warnings),
            review_result=result,
        )

        self.last_decision = decision
        self.review_write_count += 1

        return self.completed_snapshot

    @staticmethod
    def _validate_review_access(
        *,
        user_id: str,
        review_run_id: str,
    ) -> None:
        """Enforce the same opaque user boundary as production."""

        if user_id != "USER-001" or review_run_id != "EVR-API-001":
            raise CVEvidenceReviewRunUnavailableError(
                "The CV evidence review run is unavailable."
            )


def build_proposal() -> CareerEvidenceProposal:
    """Create one grounded pending CV evidence proposal."""

    source_excerpt = "Built CareerOps using Python and FastAPI."

    return CareerEvidenceProposal(
        proposal_id="EVP-001",
        category=EvidenceCategory.PROJECT,
        title="CareerOps",
        source_section=CVSection.PROJECTS,
        source_section_order_index=0,
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=["API development"],
        claims=[source_excerpt],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=source_excerpt,
            )
        ],
        warnings=[],
    )


@pytest.fixture
def workflow_service() -> FakeCVEvidenceWorkflowService:
    """Provide isolated durable review state."""

    return FakeCVEvidenceWorkflowService()


@pytest.fixture
def client(
    workflow_service: FakeCVEvidenceWorkflowService,
) -> Iterator[TestClient]:
    """Create an API client with controlled review workflow state."""

    app.dependency_overrides[get_cv_evidence_workflow_service] = lambda: (
        workflow_service
    )

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(
            get_cv_evidence_workflow_service,
            None,
        )


def test_start_evidence_review_returns_safe_awaiting_state(
    client: TestClient,
) -> None:
    """Starting review should expose pending evidence safely."""

    response = client.post(
        "/api/v1/cv-documents/DOC-001/evidence-review",
        headers={
            "X-User-ID": "USER-001",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["review_run_id"] == "EVR-API-001"
    assert body["document_id"] == "DOC-001"
    assert body["status"] == "awaiting_review"

    assert len(body["proposals"]) == 1

    assert body["proposals"][0]["proposal_id"] == "EVP-001"

    assert body["review_result"] is None

    # User ownership remains server-side only.
    assert "user_id" not in body


def test_get_evidence_review_recovers_persisted_state(
    client: TestClient,
) -> None:
    """GET should recover the same durable review state."""

    start_response = client.post(
        "/api/v1/cv-documents/DOC-001/evidence-review",
        headers={
            "X-User-ID": "USER-001",
        },
    )

    assert start_response.status_code == 200

    get_response = client.get(
        "/api/v1/cv-evidence-reviews/EVR-API-001",
        headers={
            "X-User-ID": "USER-001",
        },
    )

    assert get_response.status_code == 200

    assert get_response.json() == start_response.json()


def test_review_recovery_enforces_user_boundary(
    client: TestClient,
) -> None:
    """Another user must not learn whether a review exists."""

    response = client.get(
        "/api/v1/cv-evidence-reviews/EVR-API-001",
        headers={
            "X-User-ID": "USER-OTHER",
        },
    )

    assert response.status_code == 404

    assert response.json()["detail"] == ("The CV evidence review run is unavailable.")


def test_submit_human_review_returns_completed_evidence(
    client: TestClient,
    workflow_service: FakeCVEvidenceWorkflowService,
) -> None:
    """Explicit approval should cross the trusted evidence boundary."""

    response = client.post(
        ("/api/v1/cv-evidence-reviews/EVR-API-001/review"),
        headers={
            "X-User-ID": "USER-001",
        },
        json={"approved_proposal_ids": ["EVP-001"]},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"

    result = body["review_result"]

    assert result is not None

    assert result["approved_proposal_ids"] == ["EVP-001"]

    assert len(result["approved_evidence"]) == 1

    approved = result["approved_evidence"][0]

    assert approved["verification_status"] == ("approved")

    assert approved["approved_claims"] == ["Built CareerOps using Python and FastAPI."]

    assert approved["evidence_id"].startswith("EVD-")

    assert workflow_service.review_write_count == 1


def test_identical_http_review_retry_is_idempotent(
    client: TestClient,
    workflow_service: FakeCVEvidenceWorkflowService,
) -> None:
    """Repeating the same completed decision should not write again."""

    payload = {"approved_proposal_ids": ["EVP-001"]}

    first = client.post(
        ("/api/v1/cv-evidence-reviews/EVR-API-001/review"),
        headers={
            "X-User-ID": "USER-001",
        },
        json=payload,
    )

    second = client.post(
        ("/api/v1/cv-evidence-reviews/EVR-API-001/review"),
        headers={
            "X-User-ID": "USER-001",
        },
        json=payload,
    )

    assert first.status_code == 200
    assert second.status_code == 200

    assert second.json() == first.json()

    assert workflow_service.review_write_count == 1
