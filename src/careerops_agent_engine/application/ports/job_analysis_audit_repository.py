"""Application port for persistent job-analysis business history."""

from typing import Protocol

from careerops_agent_engine.domain.models.audit import (
    CVReviewAuditEntry,
    JobAnalysisRunSnapshot,
)


class JobAnalysisAuditRepository(Protocol):
    """Persist and retrieve business-level job-analysis history."""

    def save_run(
        self,
        snapshot: JobAnalysisRunSnapshot,
    ) -> None:
        """Create or update the latest run snapshot."""

        ...

    def save_review_result(
        self,
        *,
        snapshot: JobAnalysisRunSnapshot,
        review: CVReviewAuditEntry,
    ) -> None:
        """Atomically persist a run update and review-history entry."""

        ...

    def get_run(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> JobAnalysisRunSnapshot | None:
        """Return one user-scoped run snapshot."""

        ...

    def list_reviews(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[CVReviewAuditEntry]:
        """Return ordered human-review history for one user run."""

        ...
