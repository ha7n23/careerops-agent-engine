"""SQLAlchemy repository for persistent job-analysis audit history."""

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.application.ports.job_analysis_audit_repository import (
    JobAnalysisAuditRepository,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    JobAnalysisRunStatus,
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
    CVClaimVerificationReport,
)
from careerops_agent_engine.infrastructure.database.models.job_analysis import (
    CVReviewHistoryRecord,
    JobAnalysisRunRecord,
)


class SqlAlchemyJobAnalysisAuditRepository(JobAnalysisAuditRepository):
    """Persist business-level job-analysis history."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Store the configured SQLAlchemy session factory."""

        self._session_factory = session_factory

    def save_run(
        self,
        snapshot: JobAnalysisRunSnapshot,
    ) -> None:
        """Create or update the latest run snapshot."""

        with self._session_factory.begin() as session:
            upsert_run_record(
                session=session,
                snapshot=snapshot,
            )

    def save_review_result(
        self,
        *,
        snapshot: JobAnalysisRunSnapshot,
        review: CVReviewAuditEntry,
    ) -> None:
        """Atomically persist run state and one review-history entry."""

        if review.thread_id != snapshot.thread_id:
            raise ValueError("Review thread identifier must match the run snapshot.")

        with self._session_factory.begin() as session:
            upsert_run_record(
                session=session,
                snapshot=snapshot,
            )

            existing_review = session.get(
                CVReviewHistoryRecord,
                review.review_id,
            )

            if existing_review is not None:
                existing_domain = review_record_to_domain(existing_review)

                if not reviews_are_equivalent(
                    existing_domain,
                    review,
                ):
                    raise ValueError(
                        "Review identifier already exists with different audit data."
                    )

                # Retry-safe/idempotent replay of the same review.
                return

            existing_sequence = session.execute(
                select(CVReviewHistoryRecord).where(
                    CVReviewHistoryRecord.thread_id == review.thread_id,
                    CVReviewHistoryRecord.sequence_number == review.sequence_number,
                )
            ).scalar_one_or_none()

            if existing_sequence is not None:
                raise ValueError(
                    "Review sequence number already exists for this workflow thread."
                )

            session.add(
                CVReviewHistoryRecord(
                    review_id=review.review_id,
                    thread_id=review.thread_id,
                    sequence_number=(review.sequence_number),
                    action=review.decision.action.value,
                    decision_payload=(review.decision.model_dump(mode="json")),
                    result_status=(review.result_status.value),
                    result_review_status=(
                        review.result_review_status.value
                        if review.result_review_status is not None
                        else None
                    ),
                    resulting_cv_proposals=[
                        proposal.model_dump(mode="json")
                        for proposal in (review.resulting_cv_proposals)
                    ],
                    resulting_verification_reports=[
                        report.model_dump(mode="json")
                        for report in (review.resulting_verification_reports)
                    ],
                )
            )

    def get_run(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> JobAnalysisRunSnapshot | None:
        """Return one run while enforcing the user boundary."""

        statement = select(JobAnalysisRunRecord).where(
            JobAnalysisRunRecord.thread_id == thread_id,
            JobAnalysisRunRecord.user_id == user_id,
        )

        with self._session_factory() as session:
            record = session.execute(statement).scalar_one_or_none()

        if record is None:
            return None

        return run_record_to_domain(record)

    def list_reviews(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[CVReviewAuditEntry]:
        """Return ordered review history within the user boundary."""

        statement = (
            select(CVReviewHistoryRecord)
            .join(
                JobAnalysisRunRecord,
                CVReviewHistoryRecord.thread_id == JobAnalysisRunRecord.thread_id,
            )
            .where(
                CVReviewHistoryRecord.thread_id == thread_id,
                JobAnalysisRunRecord.user_id == user_id,
            )
            .order_by(CVReviewHistoryRecord.sequence_number)
        )

        with self._session_factory() as session:
            records = session.execute(statement).scalars().all()

        return [review_record_to_domain(record) for record in records]


def upsert_run_record(
    *,
    session: Session,
    snapshot: JobAnalysisRunSnapshot,
) -> None:
    """Insert or update the latest snapshot without changing identity."""

    record = session.get(
        JobAnalysisRunRecord,
        snapshot.thread_id,
    )

    if record is None:
        session.add(
            JobAnalysisRunRecord(
                thread_id=snapshot.thread_id,
                user_id=snapshot.user_id,
                job_id=snapshot.job_id,
                status=snapshot.status.value,
                role_title=snapshot.role_title,
                fit_score=snapshot.fit_score,
                review_status=(
                    snapshot.review_status.value
                    if snapshot.review_status is not None
                    else None
                ),
                cv_proposals=[
                    proposal.model_dump(mode="json")
                    for proposal in (snapshot.cv_proposals)
                ],
                claim_verification_reports=[
                    report.model_dump(mode="json")
                    for report in (snapshot.claim_verification_reports)
                ],
                reviewable_proposal_ids=list(snapshot.reviewable_proposal_ids),
                blocked_proposal_ids=list(snapshot.blocked_proposal_ids),
                final_cv_proposals=[
                    proposal.model_dump(mode="json")
                    for proposal in (snapshot.final_cv_proposals)
                ],
            )
        )
        return

    if record.user_id != snapshot.user_id or record.job_id != snapshot.job_id:
        raise ValueError("A job-analysis run cannot change its user or job identity.")

    record.status = snapshot.status.value
    record.role_title = snapshot.role_title
    record.fit_score = snapshot.fit_score
    record.review_status = (
        snapshot.review_status.value if snapshot.review_status is not None else None
    )

    record.cv_proposals = [
        proposal.model_dump(mode="json") for proposal in snapshot.cv_proposals
    ]

    record.claim_verification_reports = [
        report.model_dump(mode="json")
        for report in (snapshot.claim_verification_reports)
    ]

    record.reviewable_proposal_ids = list(snapshot.reviewable_proposal_ids)

    record.blocked_proposal_ids = list(snapshot.blocked_proposal_ids)

    record.final_cv_proposals = [
        proposal.model_dump(mode="json") for proposal in snapshot.final_cv_proposals
    ]


def run_record_to_domain(
    record: JobAnalysisRunRecord,
) -> JobAnalysisRunSnapshot:
    """Convert one persistent run record into the domain model."""

    return JobAnalysisRunSnapshot(
        thread_id=record.thread_id,
        user_id=record.user_id,
        job_id=record.job_id,
        status=JobAnalysisRunStatus(record.status),
        role_title=record.role_title,
        fit_score=record.fit_score,
        review_status=(
            ApprovalStatus(record.review_status)
            if record.review_status is not None
            else None
        ),
        cv_proposals=[
            CVChangeProposal.model_validate(proposal)
            for proposal in record.cv_proposals
        ],
        claim_verification_reports=[
            CVClaimVerificationReport.model_validate(report)
            for report in (record.claim_verification_reports)
        ],
        reviewable_proposal_ids=list(record.reviewable_proposal_ids),
        blocked_proposal_ids=list(record.blocked_proposal_ids),
        final_cv_proposals=[
            CVChangeProposal.model_validate(proposal)
            for proposal in record.final_cv_proposals
        ],
    )


def review_record_to_domain(
    record: CVReviewHistoryRecord,
) -> CVReviewAuditEntry:
    """Convert one persistent review record into the domain model."""

    return CVReviewAuditEntry(
        review_id=record.review_id,
        thread_id=record.thread_id,
        sequence_number=record.sequence_number,
        decision=CVReviewDecision.model_validate(record.decision_payload),
        result_status=JobAnalysisRunStatus(record.result_status),
        result_review_status=(
            ApprovalStatus(record.result_review_status)
            if record.result_review_status is not None
            else None
        ),
        resulting_cv_proposals=[
            CVChangeProposal.model_validate(proposal)
            for proposal in (record.resulting_cv_proposals)
        ],
        resulting_verification_reports=[
            CVClaimVerificationReport.model_validate(report)
            for report in (record.resulting_verification_reports)
        ],
        recorded_at=record.created_at,
    )


def reviews_are_equivalent(
    first: CVReviewAuditEntry,
    second: CVReviewAuditEntry,
) -> bool:
    """Compare retry-relevant review data while ignoring DB timestamp."""

    return first.model_dump(
        mode="json",
        exclude={"recorded_at"},
    ) == second.model_dump(
        mode="json",
        exclude={"recorded_at"},
    )
