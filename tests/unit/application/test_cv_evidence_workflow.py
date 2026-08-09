"""Tests for persistent CV evidence-review workflow orchestration."""

from hashlib import sha256

from careerops_agent_engine.application.services.cv_document_extraction import (
    CVDocumentExtractionService,
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
from careerops_agent_engine.application.services.cv_evidence_workflow import (
    CVEvidenceWorkflowService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
    EvidenceCategory,
    EvidenceOverlapScope,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ExtractedDocumentText,
    ParsedCVDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceCandidate,
    CareerEvidenceOverlapFinding,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
)
from careerops_agent_engine.infrastructure.documents.cv_section_parser import (
    DeterministicCVSectionParser,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)

DOCUMENT_BYTES = b"%PDF-1.7\nCareerOps test CV"


class FakeStorage:
    """Return controlled CV bytes for extraction."""

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Storage writes are unused in these workflow tests."""

        del user_id
        del document_id
        del document_format
        del data

        raise AssertionError("Workflow tests should not write document bytes.")

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Return the already-persisted document bytes."""

        assert user_id == "USER-001"
        assert storage_key == ("documents/usr-test/DOC-001.pdf")

        return DOCUMENT_BYTES

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Deletion is unused in these workflow tests."""

        del user_id
        del storage_key

        raise AssertionError("Workflow tests should not delete document bytes.")


class FakeNativeExtractor:
    """Return deterministic native CV text."""

    def __init__(self) -> None:
        self.call_count = 0

    def extract(
        self,
        *,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> ExtractedDocumentText:
        """Return one projects section."""

        self.call_count += 1

        assert document_format is (CareerDocumentFormat.PDF)
        assert data == DOCUMENT_BYTES

        return ExtractedDocumentText(
            document_id=document_id,
            text=("Projects\nCareerOps\nBuilt CareerOps using Python."),
            page_count=1,
            paragraph_count=None,
            warnings=[],
        )


class FakeCVEvidenceExtractor:
    """Return controlled structured evidence candidates."""

    def __init__(
        self,
        *,
        return_candidates: bool = True,
    ) -> None:
        self.return_candidates = return_candidates
        self.call_count = 0

    def extract(
        self,
        *,
        document: ParsedCVDocument,
    ) -> list[CareerEvidenceCandidate]:
        """Return one grounded project candidate when enabled."""

        self.call_count += 1

        if not self.return_candidates:
            return []

        assert document.sections
        assert document.sections[0].text == ("CareerOps\nBuilt CareerOps using Python.")

        return [
            CareerEvidenceCandidate(
                category=EvidenceCategory.PROJECT,
                title="CareerOps",
                source_section_order_index=0,
                source_excerpt=("CareerOps\nBuilt CareerOps using Python."),
                technologies=["Python"],
                capabilities=["Application development"],
                claims=[
                    "CareerOps",
                    "Built CareerOps using Python.",
                ],
                warnings=[],
            )
        ]


class FakeDocumentRepository:
    """Persist career-document metadata in memory."""

    def __init__(
        self,
        document: CareerDocument,
    ) -> None:
        self.documents: dict[
            tuple[str, str],
            CareerDocument,
        ] = {
            (
                "USER-001",
                document.document_id,
            ): document
        }

    def save(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> None:
        """Store the latest document lifecycle state."""

        self.documents[
            (
                user_id,
                document.document_id,
            )
        ] = document

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CareerDocument | None:
        """Retrieve metadata within its user boundary."""

        return self.documents.get(
            (
                user_id,
                document_id,
            )
        )


class FakeAuditRepository:
    """Persist review snapshots and audit entries in memory."""

    def __init__(self) -> None:
        self.runs: dict[
            tuple[str, str],
            CVEvidenceReviewRunSnapshot,
        ] = {}

        self.latest_by_document: dict[
            tuple[str, str],
            str,
        ] = {}

        self.reviews: dict[
            str,
            list[CVEvidenceReviewAuditEntry],
        ] = {}

        self.save_run_calls = 0
        self.save_review_result_calls = 0

    def save_run(
        self,
        snapshot: CVEvidenceReviewRunSnapshot,
    ) -> None:
        """Persist one latest non-completed snapshot."""

        self.save_run_calls += 1

        self._store_snapshot(snapshot)

    def save_review_result(
        self,
        *,
        snapshot: CVEvidenceReviewRunSnapshot,
        review: CVEvidenceReviewAuditEntry,
    ) -> None:
        """Persist the completed snapshot and review event."""

        self.save_review_result_calls += 1

        self._store_snapshot(snapshot)

        self.reviews.setdefault(
            review.review_run_id,
            [],
        ).append(review)

    def get_run(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> CVEvidenceReviewRunSnapshot | None:
        """Retrieve one run within its user boundary."""

        return self.runs.get(
            (
                user_id,
                review_run_id,
            )
        )

    def get_latest_for_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CVEvidenceReviewRunSnapshot | None:
        """Return the persisted run for one document."""

        review_run_id = self.latest_by_document.get(
            (
                user_id,
                document_id,
            )
        )

        if review_run_id is None:
            return None

        return self.get_run(
            user_id=user_id,
            review_run_id=review_run_id,
        )

    def list_reviews(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> list[CVEvidenceReviewAuditEntry]:
        """Return ordered user-owned review history."""

        if (
            user_id,
            review_run_id,
        ) not in self.runs:
            return []

        return list(
            self.reviews.get(
                review_run_id,
                [],
            )
        )

    def replace_run_for_test(
        self,
        snapshot: CVEvidenceReviewRunSnapshot,
    ) -> None:
        """Replace a snapshot to model persisted server state."""

        self._store_snapshot(snapshot)

    def _store_snapshot(
        self,
        snapshot: CVEvidenceReviewRunSnapshot,
    ) -> None:
        """Store one snapshot and its document lookup."""

        self.runs[
            (
                snapshot.user_id,
                snapshot.review_run_id,
            )
        ] = snapshot

        self.latest_by_document[
            (
                snapshot.user_id,
                snapshot.document_id,
            )
        ] = snapshot.review_run_id


def build_document() -> CareerDocument:
    """Create trusted persisted CV metadata."""

    return CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=len(DOCUMENT_BYTES),
        sha256_hex=sha256(DOCUMENT_BYTES).hexdigest(),
        storage_key=("documents/usr-test/DOC-001.pdf"),
        status=CareerDocumentStatus.UPLOADED,
    )


def build_workflow(
    *,
    return_candidates: bool = True,
) -> tuple[
    CVEvidenceWorkflowService,
    FakeDocumentRepository,
    FakeAuditRepository,
    FakeNativeExtractor,
    FakeCVEvidenceExtractor,
]:
    """Create the real workflow around controlled collaborators."""

    document_repository = FakeDocumentRepository(build_document())

    audit_repository = FakeAuditRepository()

    native_extractor = FakeNativeExtractor()

    candidate_extractor = FakeCVEvidenceExtractor(return_candidates=return_candidates)

    extraction_service = CVDocumentExtractionService(
        storage=FakeStorage(),
        extractor=native_extractor,
    )

    preparation_service = CVDocumentPreparationService(
        extraction_service=extraction_service,
        section_parser=(DeterministicCVSectionParser()),
    )

    proposal_service = CVEvidenceProposalService(extractor=candidate_extractor)

    workflow = CVEvidenceWorkflowService(
        document_repository=document_repository,
        audit_repository=audit_repository,
        preparation_service=preparation_service,
        proposal_service=proposal_service,
        duplicate_detector=(
            CVEvidenceDuplicateDetector(repository=(InMemoryEvidenceRepository()))
        ),
        review_service=(CVEvidenceReviewService()),
    )

    return (
        workflow,
        document_repository,
        audit_repository,
        native_extractor,
        candidate_extractor,
    )


def test_start_review_persists_awaiting_snapshot() -> None:
    """A new document should become a durable human-review run."""

    (
        workflow,
        document_repository,
        audit_repository,
        native_extractor,
        candidate_extractor,
    ) = build_workflow()

    snapshot = workflow.start_review(
        user_id="USER-001",
        document_id="DOC-001",
    )

    assert snapshot.status is CVEvidenceReviewRunStatus.AWAITING_REVIEW

    assert len(snapshot.proposals) == 1

    assert audit_repository.save_run_calls == 1

    assert native_extractor.call_count == 1

    assert candidate_extractor.call_count == 1

    persisted = audit_repository.get_run(
        user_id="USER-001",
        review_run_id=(snapshot.review_run_id),
    )

    assert persisted == snapshot

    document = document_repository.get(
        user_id="USER-001",
        document_id="DOC-001",
    )

    assert document is not None

    assert document.status is CareerDocumentStatus.EXTRACTED


def test_second_start_returns_persisted_run_without_reprocessing() -> None:
    """Retrying start must not rerun extraction or Gemini."""

    (
        workflow,
        _,
        audit_repository,
        native_extractor,
        candidate_extractor,
    ) = build_workflow()

    first = workflow.start_review(
        user_id="USER-001",
        document_id="DOC-001",
    )

    second = workflow.start_review(
        user_id="USER-001",
        document_id="DOC-001",
    )

    assert second == first

    assert native_extractor.call_count == 1

    assert candidate_extractor.call_count == 1

    assert audit_repository.save_run_calls == 1


def test_no_proposals_creates_invalid_durable_snapshot() -> None:
    """No grounded evidence should produce a durable invalid result."""

    (
        workflow,
        _,
        audit_repository,
        native_extractor,
        candidate_extractor,
    ) = build_workflow(return_candidates=False)

    snapshot = workflow.start_review(
        user_id="USER-001",
        document_id="DOC-001",
    )

    assert snapshot.status is CVEvidenceReviewRunStatus.INVALID

    assert snapshot.proposals == []

    assert any(
        "No grounded career evidence" in warning
        for warning in snapshot.document_warnings
    )

    assert native_extractor.call_count == 1

    assert candidate_extractor.call_count == 1

    assert audit_repository.save_run_calls == 1


def test_submit_review_uses_persisted_proposals_and_findings() -> None:
    """Human approval should operate on the durable review snapshot."""

    (
        workflow,
        _,
        audit_repository,
        native_extractor,
        candidate_extractor,
    ) = build_workflow()

    awaiting = workflow.start_review(
        user_id="USER-001",
        document_id="DOC-001",
    )

    proposal = awaiting.proposals[0]

    finding = CareerEvidenceOverlapFinding(
        proposal_id=proposal.proposal_id,
        scope=(EvidenceOverlapScope.APPROVED_EVIDENCE),
        matching_evidence_id="EVD-EXISTING",
        matched_claims=[proposal.claims[0]],
        same_source_excerpt=False,
    )

    persisted_with_finding = awaiting.model_copy(update={"overlap_findings": [finding]})

    audit_repository.replace_run_for_test(persisted_with_finding)

    native_calls_before = native_extractor.call_count

    candidate_calls_before = candidate_extractor.call_count

    decision = EvidenceReviewDecision(
        approved_proposal_ids=[proposal.proposal_id],
        acknowledged_overlap_proposal_ids=[proposal.proposal_id],
        reviewer_comment=("Reviewed and accepted overlap."),
    )

    completed = workflow.submit_review(
        user_id="USER-001",
        review_run_id=(awaiting.review_run_id),
        decision=decision,
    )

    assert completed.status is CVEvidenceReviewRunStatus.COMPLETED

    assert completed.review_result is not None

    assert len(completed.review_result.approved_evidence) == 1

    approved = completed.review_result.approved_evidence[0]

    assert approved.verification_status is VerificationStatus.APPROVED

    assert native_extractor.call_count == native_calls_before

    assert candidate_extractor.call_count == candidate_calls_before

    assert audit_repository.save_review_result_calls == 1

    reviews = audit_repository.list_reviews(
        user_id="USER-001",
        review_run_id=(awaiting.review_run_id),
    )

    assert len(reviews) == 1
    assert reviews[0].decision == decision


def test_identical_completed_retry_does_not_write_second_review() -> None:
    """An exact client retry should return the completed snapshot."""

    (
        workflow,
        _,
        audit_repository,
        _,
        _,
    ) = build_workflow()

    awaiting = workflow.start_review(
        user_id="USER-001",
        document_id="DOC-001",
    )

    decision = EvidenceReviewDecision(
        approved_proposal_ids=[awaiting.proposals[0].proposal_id]
    )

    first = workflow.submit_review(
        user_id="USER-001",
        review_run_id=(awaiting.review_run_id),
        decision=decision,
    )

    second = workflow.submit_review(
        user_id="USER-001",
        review_run_id=(awaiting.review_run_id),
        decision=decision,
    )

    assert second == first

    assert audit_repository.save_review_result_calls == 1

    assert (
        len(
            audit_repository.list_reviews(
                user_id="USER-001",
                review_run_id=(awaiting.review_run_id),
            )
        )
        == 1
    )
