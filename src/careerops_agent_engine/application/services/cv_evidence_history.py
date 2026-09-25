"""Application queries for CV upload and evidence-review history."""

from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentHistoryRepository,
)
from careerops_agent_engine.application.ports.cv_evidence_audit_repository import (
    CVEvidenceReviewHistoryRepository,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocumentSummary,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewRunSummary,
)

DEFAULT_HISTORY_LIMIT = 50
MAX_HISTORY_LIMIT = 100


class CVEvidenceHistoryService:
    """Expose bounded user-owned CV evidence workflow history."""

    def __init__(
        self,
        *,
        document_repository: CareerDocumentHistoryRepository,
        review_repository: CVEvidenceReviewHistoryRepository,
    ) -> None:
        """Store the history query repositories."""

        self._document_repository = document_repository
        self._review_repository = review_repository

    def list_documents(
        self,
        *,
        user_id: str,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[CareerDocumentSummary]:
        """Return the user's newest uploaded documents."""

        validate_history_limit(limit)

        return self._document_repository.list_summaries(
            user_id=user_id,
            limit=limit,
        )

    def list_review_runs(
        self,
        *,
        user_id: str,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[CVEvidenceReviewRunSummary]:
        """Return the user's newest evidence-review runs."""

        validate_history_limit(limit)

        return self._review_repository.list_run_summaries(
            user_id=user_id,
            limit=limit,
        )


def validate_history_limit(limit: int) -> None:
    """Reject unbounded or meaningless history queries."""

    if not 1 <= limit <= MAX_HISTORY_LIMIT:
        raise ValueError(f"History limit must be between 1 and {MAX_HISTORY_LIMIT}.")
