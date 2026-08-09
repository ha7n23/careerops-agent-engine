"""Tests for transactional CV evidence-review persistence."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceProposal,
    SourceReference,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
    EvidenceReviewResult,
)
from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_career_documents import (  # noqa: E501
    SqlAlchemyCareerDocumentRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_cv_evidence_audit import (  # noqa: E501
    SqlAlchemyCVEvidenceAuditRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)


def build_document() -> CareerDocument:
    """Create one persisted CV document."""

    return CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key=("documents/usr-test/DOC-001.pdf"),
        status=CareerDocumentStatus.UPLOADED,
    )


def build_proposal() -> CareerEvidenceProposal:
    """Create one pending evidence proposal."""

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
        claims=["Built CareerOps using Python and FastAPI."],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=("Built CareerOps using Python and FastAPI."),
            )
        ],
        warnings=[],
    )


def build_approved_evidence() -> CareerEvidence:
    """Create the trusted result of human approval."""

    return CareerEvidence(
        evidence_id="EVD-001",
        category=EvidenceCategory.PROJECT,
        title="CareerOps",
        verification_status=(VerificationStatus.APPROVED),
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=["API development"],
        approved_claims=["Built CareerOps using Python and FastAPI."],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=("Built CareerOps using Python and FastAPI."),
            )
        ],
    )


def build_completed_result() -> EvidenceReviewResult:
    """Create one accepted human-review result."""

    return EvidenceReviewResult(
        approved_proposal_ids=["EVP-001"],
        approved_evidence=[build_approved_evidence()],
    )


def build_awaiting_snapshot() -> CVEvidenceReviewRunSnapshot:
    """Create the initial durable review snapshot."""

    return CVEvidenceReviewRunSnapshot(
        review_run_id="EVR-001",
        user_id="USER-001",
        document_id="DOC-001",
        status=(CVEvidenceReviewRunStatus.AWAITING_REVIEW),
        proposals=[build_proposal()],
        overlap_findings=[],
        document_warnings=[],
    )


def build_completed_snapshot(
    *,
    warnings: list[str] | None = None,
) -> CVEvidenceReviewRunSnapshot:
    """Create the completed business snapshot."""

    return CVEvidenceReviewRunSnapshot(
        review_run_id="EVR-001",
        user_id="USER-001",
        document_id="DOC-001",
        status=(CVEvidenceReviewRunStatus.COMPLETED),
        proposals=[build_proposal()],
        overlap_findings=[],
        document_warnings=(list(warnings) if warnings is not None else []),
        review_result=build_completed_result(),
    )


def build_review(
    *,
    review_id: str = "EVR-AUDIT-001",
    sequence_number: int = 1,
) -> CVEvidenceReviewAuditEntry:
    """Create one applied human-review audit event."""

    return CVEvidenceReviewAuditEntry(
        review_id=review_id,
        review_run_id="EVR-001",
        sequence_number=sequence_number,
        decision=EvidenceReviewDecision(approved_proposal_ids=["EVP-001"]),
        result=build_completed_result(),
    )


@pytest.fixture
def repositories() -> Iterator[
    tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ]
]:
    """Create repositories sharing one isolated database."""

    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = create_session_factory(engine)

    document_repository = SqlAlchemyCareerDocumentRepository(session_factory)

    document_repository.save(
        user_id="USER-001",
        document=build_document(),
    )

    try:
        yield (
            SqlAlchemyCVEvidenceAuditRepository(session_factory),
            document_repository,
            SqlAlchemyEvidenceRepository(session_factory),
        )
    finally:
        engine.dispose()


def test_save_and_get_awaiting_run(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """The exact proposals shown to humans should survive storage."""

    audit_repository, _, _ = repositories

    snapshot = build_awaiting_snapshot()

    audit_repository.save_run(snapshot)

    assert (
        audit_repository.get_run(
            user_id="USER-001",
            review_run_id="EVR-001",
        )
        == snapshot
    )

    assert (
        audit_repository.get_run(
            user_id="USER-OTHER",
            review_run_id="EVR-001",
        )
        is None
    )


def test_completed_run_cannot_bypass_review_transaction(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """Completed state must include history and evidence atomically."""

    audit_repository, _, _ = repositories

    with pytest.raises(
        ValueError,
        match="must be persisted with save_review_result",
    ):
        audit_repository.save_run(build_completed_snapshot())


def test_review_transaction_persists_run_history_and_evidence(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """One transaction should cross the full evidence trust boundary."""

    (
        audit_repository,
        _,
        evidence_repository,
    ) = repositories

    snapshot = build_completed_snapshot()
    review = build_review()

    audit_repository.save_review_result(
        snapshot=snapshot,
        review=review,
    )

    assert (
        audit_repository.get_run(
            user_id="USER-001",
            review_run_id="EVR-001",
        )
        == snapshot
    )

    reviews = audit_repository.list_reviews(
        user_id="USER-001",
        review_run_id="EVR-001",
    )

    assert len(reviews) == 1
    assert reviews[0].review_id == ("EVR-AUDIT-001")
    assert reviews[0].recorded_at is not None

    approved = evidence_repository.get_approved(
        user_id="USER-001",
        evidence_id="EVD-001",
    )

    assert approved == build_approved_evidence()


def test_identical_review_retry_is_idempotent(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """Retrying the same committed human decision changes nothing."""

    audit_repository, _, _ = repositories

    snapshot = build_completed_snapshot()
    review = build_review()

    audit_repository.save_review_result(
        snapshot=snapshot,
        review=review,
    )

    audit_repository.save_review_result(
        snapshot=snapshot,
        review=review,
    )

    assert (
        len(
            audit_repository.list_reviews(
                user_id="USER-001",
                review_run_id="EVR-001",
            )
        )
        == 1
    )


def test_duplicate_sequence_rolls_back_snapshot_update(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """A failed history append must roll back the run mutation."""

    audit_repository, _, _ = repositories

    original_snapshot = build_completed_snapshot()

    audit_repository.save_review_result(
        snapshot=original_snapshot,
        review=build_review(),
    )

    changed_snapshot = build_completed_snapshot(warnings=["Must not persist"])

    with pytest.raises(
        ValueError,
        match="sequence number already exists",
    ):
        audit_repository.save_review_result(
            snapshot=changed_snapshot,
            review=build_review(
                review_id="EVR-AUDIT-002",
                sequence_number=1,
            ),
        )

    assert (
        audit_repository.get_run(
            user_id="USER-001",
            review_run_id="EVR-001",
        )
        == original_snapshot
    )


def test_evidence_collision_rolls_back_run_and_history(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """Approved-evidence failure must roll back all business writes."""

    (
        audit_repository,
        _,
        evidence_repository,
    ) = repositories

    original_snapshot = build_completed_snapshot()

    audit_repository.save_review_result(
        snapshot=original_snapshot,
        review=build_review(),
    )

    changed_snapshot = build_completed_snapshot(warnings=["Must also roll back"])

    with pytest.raises(
        ValueError,
        match="Approved evidence identifier already exists",
    ):
        audit_repository.save_review_result(
            snapshot=changed_snapshot,
            review=build_review(
                review_id="EVR-AUDIT-002",
                sequence_number=2,
            ),
        )

    assert (
        audit_repository.get_run(
            user_id="USER-001",
            review_run_id="EVR-001",
        )
        == original_snapshot
    )

    assert (
        len(
            audit_repository.list_reviews(
                user_id="USER-001",
                review_run_id="EVR-001",
            )
        )
        == 1
    )

    assert (
        evidence_repository.get_approved(
            user_id="USER-001",
            evidence_id="EVD-001",
        )
        == build_approved_evidence()
    )


def test_review_history_is_user_scoped(
    repositories: tuple[
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """Audit history must remain inside the authenticated user scope."""

    audit_repository, _, _ = repositories

    audit_repository.save_review_result(
        snapshot=build_completed_snapshot(),
        review=build_review(),
    )

    assert (
        audit_repository.list_reviews(
            user_id="USER-OTHER",
            review_run_id="EVR-001",
        )
        == []
    )
