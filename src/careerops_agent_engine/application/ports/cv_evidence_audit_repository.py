"""Application port for persistent CV evidence-review history."""

from typing import Protocol

from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)


class CVEvidenceAuditRepository(Protocol):
    """Persist the CV evidence-review business workflow."""

    def save_run(
        self,
        snapshot: CVEvidenceReviewRunSnapshot,
    ) -> None:
        """Persist a non-completed latest review snapshot."""

        ...

    def save_review_result(
        self,
        *,
        snapshot: CVEvidenceReviewRunSnapshot,
        review: CVEvidenceReviewAuditEntry,
    ) -> None:
        """Atomically persist review state, history and approved evidence."""

        ...

    def get_run(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> CVEvidenceReviewRunSnapshot | None:
        """Retrieve one user-owned evidence-review run."""

        ...

    def get_latest_for_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CVEvidenceReviewRunSnapshot | None:
        """Retrieve the latest persisted review run for one document."""

        ...

    def list_reviews(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> list[CVEvidenceReviewAuditEntry]:
        """Return ordered review history for one user-owned run."""

        ...
