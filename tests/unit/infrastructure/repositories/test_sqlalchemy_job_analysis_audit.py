"""Tests for the SQLAlchemy job-analysis audit repository."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVSection,
    JobAnalysisRunStatus,
    ReviewAction,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)
from careerops_agent_engine.domain.models.audit import (
    CVReviewAuditEntry,
    JobAnalysisRunSnapshot,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.verification import (
    ClaimAssessment,
    CVClaimVerificationReport,
)
from careerops_agent_engine.infrastructure.database.base import (
    Base,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
)
from careerops_agent_engine.infrastructure.repositories import (
    sqlalchemy_job_analysis_audit,
)


def build_proposal() -> CVChangeProposal:
    """Create one grounded proposal for persistence tests."""

    return CVChangeProposal(
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        proposed_text=("Built a Python API using FastAPI."),
        requirement_ids=["REQ-001"],
        supporting_evidence_ids=["EVD-001"],
        confidence_score=0.95,
        warnings=[],
    )


def build_report() -> CVClaimVerificationReport:
    """Create one supported claim-verification report."""

    return CVClaimVerificationReport(
        proposal_id="CVP-001",
        claims=[
            ClaimAssessment(
                claim_text=("Built a Python API using FastAPI."),
                supported=True,
                supporting_evidence_ids=["EVD-001"],
                explanation=(
                    "Approved evidence directly supports the Python and FastAPI claim."
                ),
            )
        ],
        coverage_complete=True,
        coverage_notes=[],
        fully_supported=True,
        unsupported_claims=[],
    )


def build_snapshot(
    *,
    status: JobAnalysisRunStatus = (JobAnalysisRunStatus.AWAITING_REVIEW),
    role_title: str = "Junior AI Engineer",
    review_status: ApprovalStatus | None = None,
) -> JobAnalysisRunSnapshot:
    """Create one business run snapshot."""

    proposal = build_proposal()

    return JobAnalysisRunSnapshot(
        thread_id="THR-001",
        user_id="USER-001",
        job_id="JOB-001",
        status=status,
        role_title=role_title,
        fit_score=88.0,
        review_status=review_status,
        cv_proposals=[proposal],
        claim_verification_reports=[build_report()],
        reviewable_proposal_ids=(
            ["CVP-001"] if status is JobAnalysisRunStatus.AWAITING_REVIEW else []
        ),
        blocked_proposal_ids=[],
        final_cv_proposals=(
            [proposal] if status is JobAnalysisRunStatus.COMPLETED else []
        ),
    )


def build_review(
    *,
    review_id: str = "REV-001",
    sequence_number: int = 1,
    action: ReviewAction = ReviewAction.APPROVE,
    result_status: JobAnalysisRunStatus = (JobAnalysisRunStatus.COMPLETED),
) -> CVReviewAuditEntry:
    """Create one persistent human-review event."""

    if action is ReviewAction.APPROVE:
        decision = CVReviewDecision(
            action=ReviewAction.APPROVE,
            approved_proposal_ids=["CVP-001"],
        )
        result_review_status = ApprovalStatus.APPROVED
    else:
        decision = CVReviewDecision(
            action=ReviewAction.REGENERATE,
            rejected_proposal_ids=["CVP-001"],
            reviewer_comment=("Make the wording more concise."),
        )
        result_review_status = None

    return CVReviewAuditEntry(
        review_id=review_id,
        thread_id="THR-001",
        sequence_number=sequence_number,
        decision=decision,
        result_status=result_status,
        result_review_status=(result_review_status),
        resulting_cv_proposals=[build_proposal()],
        resulting_verification_reports=[build_report()],
    )


@pytest.fixture
def repository() -> Iterator[
    sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository
]:
    """Create an isolated SQLite-backed audit repository."""

    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = create_session_factory(engine)

    try:
        yield (
            sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository(
                session_factory
            )
        )
    finally:
        engine.dispose()


def test_save_and_get_run_round_trip(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """Latest business state should round-trip through storage."""

    snapshot = build_snapshot()

    repository.save_run(snapshot)

    loaded = repository.get_run(
        user_id="USER-001",
        thread_id="THR-001",
    )

    assert loaded == snapshot


def test_get_run_enforces_user_boundary(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """Another user must not access the stored run."""

    repository.save_run(build_snapshot())

    loaded = repository.get_run(
        user_id="USER-OTHER",
        thread_id="THR-001",
    )

    assert loaded is None


def test_save_run_updates_latest_snapshot(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """Saving the same run should update rather than duplicate it."""

    repository.save_run(build_snapshot())

    repository.save_run(
        build_snapshot(
            status=JobAnalysisRunStatus.COMPLETED,
            role_title="Python Engineer",
            review_status=ApprovalStatus.APPROVED,
        )
    )

    loaded = repository.get_run(
        user_id="USER-001",
        thread_id="THR-001",
    )

    assert loaded is not None
    assert loaded.status is JobAnalysisRunStatus.COMPLETED
    assert loaded.role_title == "Python Engineer"
    assert loaded.review_status is ApprovalStatus.APPROVED
    assert loaded.final_cv_proposals


def test_save_review_result_persists_snapshot_and_history(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """One transaction should persist the result and review event."""

    completed_snapshot = build_snapshot(
        status=JobAnalysisRunStatus.COMPLETED,
        review_status=ApprovalStatus.APPROVED,
    )

    review = build_review()

    repository.save_review_result(
        snapshot=completed_snapshot,
        review=review,
    )

    loaded_run = repository.get_run(
        user_id="USER-001",
        thread_id="THR-001",
    )

    reviews = repository.list_reviews(
        user_id="USER-001",
        thread_id="THR-001",
    )

    assert loaded_run == completed_snapshot
    assert len(reviews) == 1

    stored_review = reviews[0]

    assert stored_review.review_id == "REV-001"
    assert stored_review.sequence_number == 1
    assert stored_review.decision.action is ReviewAction.APPROVE
    assert stored_review.result_status is JobAnalysisRunStatus.COMPLETED
    assert stored_review.recorded_at is not None


def test_same_review_retry_is_idempotent(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """Retrying the identical review must not duplicate history."""

    snapshot = build_snapshot(
        status=JobAnalysisRunStatus.COMPLETED,
        review_status=ApprovalStatus.APPROVED,
    )
    review = build_review()

    repository.save_review_result(
        snapshot=snapshot,
        review=review,
    )

    repository.save_review_result(
        snapshot=snapshot,
        review=review,
    )

    reviews = repository.list_reviews(
        user_id="USER-001",
        thread_id="THR-001",
    )

    assert len(reviews) == 1


def test_duplicate_sequence_rolls_back_snapshot_update(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """A failed review append must also roll back the run update."""

    first_snapshot = build_snapshot(
        role_title="Original title",
    )

    first_review = build_review(
        review_id="REV-001",
        sequence_number=1,
    )

    repository.save_review_result(
        snapshot=first_snapshot,
        review=first_review,
    )

    conflicting_snapshot = build_snapshot(
        role_title="Must not persist",
    )

    conflicting_review = build_review(
        review_id="REV-002",
        sequence_number=1,
    )

    with pytest.raises(
        ValueError,
        match="sequence number already exists",
    ):
        repository.save_review_result(
            snapshot=conflicting_snapshot,
            review=conflicting_review,
        )

    loaded = repository.get_run(
        user_id="USER-001",
        thread_id="THR-001",
    )

    assert loaded is not None
    assert loaded.role_title == "Original title"


def test_list_reviews_is_user_scoped_and_sequence_ordered(
    repository: sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository,
) -> None:
    """History should remain authenticated and chronologically ordered."""

    awaiting_snapshot = build_snapshot()

    second_review = build_review(
        review_id="REV-002",
        sequence_number=2,
        action=ReviewAction.REGENERATE,
        result_status=(JobAnalysisRunStatus.AWAITING_REVIEW),
    )

    first_review = build_review(
        review_id="REV-001",
        sequence_number=1,
        action=ReviewAction.REGENERATE,
        result_status=(JobAnalysisRunStatus.AWAITING_REVIEW),
    )

    repository.save_review_result(
        snapshot=awaiting_snapshot,
        review=second_review,
    )

    repository.save_review_result(
        snapshot=awaiting_snapshot,
        review=first_review,
    )

    reviews = repository.list_reviews(
        user_id="USER-001",
        thread_id="THR-001",
    )

    assert [review.sequence_number for review in reviews] == [1, 2]

    assert (
        repository.list_reviews(
            user_id="USER-OTHER",
            thread_id="THR-001",
        )
        == []
    )
