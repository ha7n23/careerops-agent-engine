"""Persistent orchestration for CV evidence extraction and human review."""

from hashlib import sha256
from uuid import uuid4

from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    CVEvidenceReviewRunUnavailableError,
)
from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.application.ports.cv_evidence_audit_repository import (
    CVEvidenceAuditRepository,
)
from careerops_agent_engine.application.services.cv_document_preparation import (
    CVDocumentPreparationService,
)
from careerops_agent_engine.application.services.cv_evidence_duplicates import (
    CVEvidenceDuplicateDetector,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    CVEvidenceProposalService,
)
from careerops_agent_engine.application.services.cv_evidence_review import (
    CVEvidenceReviewService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
)


class CVEvidenceWorkflowService:
    """Coordinate durable CV evidence extraction and approval."""

    def __init__(
        self,
        *,
        document_repository: CareerDocumentRepository,
        audit_repository: CVEvidenceAuditRepository,
        preparation_service: CVDocumentPreparationService,
        proposal_service: CVEvidenceProposalService,
        duplicate_detector: CVEvidenceDuplicateDetector,
        review_service: CVEvidenceReviewService,
    ) -> None:
        """Store workflow dependencies."""

        self._document_repository = document_repository
        self._audit_repository = audit_repository
        self._preparation_service = preparation_service
        self._proposal_service = proposal_service
        self._duplicate_detector = duplicate_detector
        self._review_service = review_service

    def start_review(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CVEvidenceReviewRunSnapshot:
        """Create or recover the durable evidence-review snapshot."""

        existing = self._audit_repository.get_latest_for_document(
            user_id=user_id,
            document_id=document_id,
        )

        if existing is not None:
            return existing

        document = self._document_repository.get(
            user_id=user_id,
            document_id=document_id,
        )

        if document is None or document.status is CareerDocumentStatus.QUARANTINED:
            raise CareerDocumentUnavailableError("The career document is unavailable.")

        parsed = self._preparation_service.prepare(
            user_id=user_id,
            document=document,
        )

        extracted_document = document.model_copy(
            update={"status": (CareerDocumentStatus.EXTRACTED)}
        )

        self._document_repository.save(
            user_id=user_id,
            document=extracted_document,
        )

        proposals = self._proposal_service.generate(document=parsed)

        review_run_id = build_review_run_id()

        if not proposals:
            snapshot = CVEvidenceReviewRunSnapshot(
                review_run_id=review_run_id,
                user_id=user_id,
                document_id=document_id,
                status=(CVEvidenceReviewRunStatus.INVALID),
                proposals=[],
                overlap_findings=[],
                document_warnings=[
                    *parsed.warnings,
                    ("No grounded career evidence proposals were extracted."),
                ],
            )

            self._audit_repository.save_run(snapshot)

            return snapshot

        overlap_findings = self._duplicate_detector.detect(
            user_id=user_id,
            proposals=proposals,
        )

        snapshot = CVEvidenceReviewRunSnapshot(
            review_run_id=review_run_id,
            user_id=user_id,
            document_id=document_id,
            status=(CVEvidenceReviewRunStatus.AWAITING_REVIEW),
            proposals=proposals,
            overlap_findings=overlap_findings,
            document_warnings=list(parsed.warnings),
        )

        self._audit_repository.save_run(snapshot)

        return snapshot

    def get_review(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> CVEvidenceReviewRunSnapshot:
        """Retrieve one persisted review run inside its user boundary."""

        snapshot = self._audit_repository.get_run(
            user_id=user_id,
            review_run_id=review_run_id,
        )

        if snapshot is None:
            raise CVEvidenceReviewRunUnavailableError(
                "The CV evidence review run is unavailable."
            )

        return snapshot

    def submit_review(
        self,
        *,
        user_id: str,
        review_run_id: str,
        decision: EvidenceReviewDecision,
    ) -> CVEvidenceReviewRunSnapshot:
        """Apply human review using only the persisted review snapshot."""

        snapshot = self._audit_repository.get_run(
            user_id=user_id,
            review_run_id=review_run_id,
        )

        if snapshot is None:
            raise CVEvidenceReviewRunUnavailableError(
                "The CV evidence review run is unavailable."
            )

        if snapshot.status is CVEvidenceReviewRunStatus.COMPLETED:
            return validate_completed_retry(
                audit_repository=self._audit_repository,
                snapshot=snapshot,
                decision=decision,
            )

        if snapshot.status is not CVEvidenceReviewRunStatus.AWAITING_REVIEW:
            raise CVEvidenceReviewRunUnavailableError(
                "The CV evidence review run is unavailable."
            )

        result = self._review_service.review(
            proposals=snapshot.proposals,
            overlap_findings=(snapshot.overlap_findings),
            decision=decision,
        )

        completed_snapshot = CVEvidenceReviewRunSnapshot(
            review_run_id=(snapshot.review_run_id),
            user_id=snapshot.user_id,
            document_id=snapshot.document_id,
            status=(CVEvidenceReviewRunStatus.COMPLETED),
            proposals=list(snapshot.proposals),
            overlap_findings=list(snapshot.overlap_findings),
            document_warnings=list(snapshot.document_warnings),
            review_result=result,
        )

        existing_reviews = self._audit_repository.list_reviews(
            user_id=user_id,
            review_run_id=review_run_id,
        )

        sequence_number = (
            max(
                (review.sequence_number for review in existing_reviews),
                default=0,
            )
            + 1
        )

        review = CVEvidenceReviewAuditEntry(
            review_id=build_review_id(
                review_run_id=review_run_id,
                sequence_number=sequence_number,
            ),
            review_run_id=review_run_id,
            sequence_number=sequence_number,
            decision=decision,
            result=result,
        )

        self._audit_repository.save_review_result(
            snapshot=completed_snapshot,
            review=review,
        )

        return completed_snapshot


def build_review_run_id() -> str:
    """Create an opaque evidence-review run identifier."""

    return f"EVR-{uuid4().hex.upper()}"


def build_review_id(
    *,
    review_run_id: str,
    sequence_number: int,
) -> str:
    """Create a stable audit identifier for one review sequence."""

    fingerprint = f"{review_run_id}:{sequence_number}"

    digest = sha256(fingerprint.encode("utf-8")).hexdigest()[:16].upper()

    return f"EVH-{digest}"


def validate_completed_retry(
    *,
    audit_repository: CVEvidenceAuditRepository,
    snapshot: CVEvidenceReviewRunSnapshot,
    decision: EvidenceReviewDecision,
) -> CVEvidenceReviewRunSnapshot:
    """Allow exact client retries without creating another review."""

    reviews = audit_repository.list_reviews(
        user_id=snapshot.user_id,
        review_run_id=snapshot.review_run_id,
    )

    if (
        reviews
        and reviews[-1].decision == decision
        and snapshot.review_result == reviews[-1].result
    ):
        return snapshot

    raise CVEvidenceReviewRunUnavailableError(
        "The CV evidence review run is unavailable."
    )
