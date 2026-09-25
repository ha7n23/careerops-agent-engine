"""SQLAlchemy persistence for CV evidence-review business history."""

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.application.ports.cv_evidence_audit_repository import (
    CVEvidenceAuditRepository,
    CVEvidenceReviewHistoryRepository,
)
from careerops_agent_engine.domain.enums import (
    CVEvidenceReviewRunStatus,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
    CVEvidenceReviewRunSummary,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
    EvidenceReviewResult,
)
from careerops_agent_engine.infrastructure.database.models.cv_evidence import (
    CareerDocumentRecord,
    CVEvidenceReviewHistoryRecord,
    CVEvidenceReviewRunRecord,
)
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceRecord,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    record_to_domain,
)


class SqlAlchemyCVEvidenceAuditRepository(
    CVEvidenceAuditRepository,
    CVEvidenceReviewHistoryRepository,
):
    """Persist review state and approved evidence atomically."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Store the configured SQLAlchemy session factory."""

        self._session_factory = session_factory

    def save_run(
        self,
        snapshot: CVEvidenceReviewRunSnapshot,
    ) -> None:
        """Persist a review snapshot before final human approval."""

        if snapshot.status is CVEvidenceReviewRunStatus.COMPLETED:
            raise ValueError(
                "Completed CV evidence review runs "
                "must be persisted with save_review_result."
            )

        with self._session_factory.begin() as session:
            upsert_review_run(
                session=session,
                snapshot=snapshot,
            )

    def save_review_result(
        self,
        *,
        snapshot: CVEvidenceReviewRunSnapshot,
        review: CVEvidenceReviewAuditEntry,
    ) -> None:
        """Commit snapshot, audit event and approved evidence together."""

        validate_completed_review(
            snapshot=snapshot,
            review=review,
        )

        with self._session_factory.begin() as session:
            existing_review = session.get(
                CVEvidenceReviewHistoryRecord,
                review.review_id,
            )

            if existing_review is not None:
                validate_idempotent_review_retry(
                    session=session,
                    snapshot=snapshot,
                    review=review,
                    existing_review=existing_review,
                )
                return

            upsert_review_run(
                session=session,
                snapshot=snapshot,
            )

            # Ensure a newly created parent review-run row exists in the
            # database before inserting child history rows that reference it.
            session.flush()

            existing_sequence = session.execute(
                select(CVEvidenceReviewHistoryRecord).where(
                    CVEvidenceReviewHistoryRecord.review_run_id == review.review_run_id,
                    CVEvidenceReviewHistoryRecord.sequence_number
                    == review.sequence_number,
                )
            ).scalar_one_or_none()

            if existing_sequence is not None:
                raise ValueError(
                    "Review sequence number already exists "
                    "for this CV evidence review run."
                )

            for evidence in review.result.approved_evidence:
                insert_approved_evidence(
                    session=session,
                    user_id=snapshot.user_id,
                    evidence=evidence,
                )

            session.add(
                CVEvidenceReviewHistoryRecord(
                    review_id=review.review_id,
                    review_run_id=review.review_run_id,
                    sequence_number=review.sequence_number,
                    decision_payload=(review.decision.model_dump(mode="json")),
                    result_payload=(review.result.model_dump(mode="json")),
                )
            )

    def get_run(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> CVEvidenceReviewRunSnapshot | None:
        """Retrieve one review run inside its user boundary."""

        statement = select(CVEvidenceReviewRunRecord).where(
            CVEvidenceReviewRunRecord.review_run_id == review_run_id,
            CVEvidenceReviewRunRecord.user_id == user_id,
        )

        with self._session_factory() as session:
            record = session.execute(statement).scalar_one_or_none()

        if record is None:
            return None

        return review_run_record_to_domain(record)

    def list_reviews(
        self,
        *,
        user_id: str,
        review_run_id: str,
    ) -> list[CVEvidenceReviewAuditEntry]:
        """Return ordered history inside the authenticated boundary."""

        statement = (
            select(CVEvidenceReviewHistoryRecord)
            .join(
                CVEvidenceReviewRunRecord,
                (
                    CVEvidenceReviewHistoryRecord.review_run_id
                    == CVEvidenceReviewRunRecord.review_run_id
                ),
            )
            .where(
                CVEvidenceReviewHistoryRecord.review_run_id == review_run_id,
                CVEvidenceReviewRunRecord.user_id == user_id,
            )
            .order_by(CVEvidenceReviewHistoryRecord.sequence_number)
        )

        with self._session_factory() as session:
            records = session.execute(statement).scalars().all()

        return [review_history_record_to_domain(record) for record in records]

    def get_latest_for_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CVEvidenceReviewRunSnapshot | None:
        """Retrieve the newest persisted review run for one document."""

        statement = (
            select(CVEvidenceReviewRunRecord)
            .where(
                CVEvidenceReviewRunRecord.user_id == user_id,
                CVEvidenceReviewRunRecord.document_id == document_id,
            )
            .order_by(
                CVEvidenceReviewRunRecord.created_at.desc(),
                CVEvidenceReviewRunRecord.review_run_id.desc(),
            )
            .limit(1)
        )

        with self._session_factory() as session:
            record = session.execute(statement).scalar_one_or_none()

        if record is None:
            return None

        return review_run_record_to_domain(record)

    def list_run_summaries(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[CVEvidenceReviewRunSummary]:
        """Return bounded review-run history inside the user boundary."""

        statement = (
            select(CVEvidenceReviewRunRecord)
            .where(CVEvidenceReviewRunRecord.user_id == user_id)
            .order_by(
                CVEvidenceReviewRunRecord.created_at.desc(),
                CVEvidenceReviewRunRecord.review_run_id.desc(),
            )
            .limit(limit)
        )

        with self._session_factory() as session:
            records = session.execute(statement).scalars().all()

        return [review_run_record_to_summary(record) for record in records]


def validate_completed_review(
    *,
    snapshot: CVEvidenceReviewRunSnapshot,
    review: CVEvidenceReviewAuditEntry,
) -> None:
    """Validate the final persistence boundary."""

    if review.review_run_id != snapshot.review_run_id:
        raise ValueError(
            "Review-run identifier must match the CV evidence run snapshot."
        )

    if snapshot.status is not CVEvidenceReviewRunStatus.COMPLETED:
        raise ValueError(
            "A persisted human review result requires a completed run snapshot."
        )

    if snapshot.review_result != review.result:
        raise ValueError(
            "Run snapshot review result must match the audit review result."
        )


def upsert_review_run(
    *,
    session: Session,
    snapshot: CVEvidenceReviewRunSnapshot,
) -> None:
    """Insert or update one latest business snapshot."""

    owned_document = session.get(
        CareerDocumentRecord,
        (
            snapshot.user_id,
            snapshot.document_id,
        ),
    )

    if owned_document is None:
        raise ValueError(
            "CV evidence review requires an existing user-owned career document."
        )

    record = session.get(
        CVEvidenceReviewRunRecord,
        snapshot.review_run_id,
    )

    if record is None:
        session.add(
            CVEvidenceReviewRunRecord(
                review_run_id=snapshot.review_run_id,
                user_id=snapshot.user_id,
                document_id=snapshot.document_id,
                status=snapshot.status.value,
                proposals=[
                    proposal.model_dump(mode="json") for proposal in snapshot.proposals
                ],
                overlap_findings=[
                    finding.model_dump(mode="json")
                    for finding in snapshot.overlap_findings
                ],
                document_warnings=list(snapshot.document_warnings),
                review_result=(
                    snapshot.review_result.model_dump(mode="json")
                    if snapshot.review_result is not None
                    else None
                ),
            )
        )
        return

    if record.user_id != snapshot.user_id or record.document_id != snapshot.document_id:
        raise ValueError(
            "A CV evidence review run cannot change its user or document identity."
        )

    record.status = snapshot.status.value

    record.proposals = [
        proposal.model_dump(mode="json") for proposal in snapshot.proposals
    ]

    record.overlap_findings = [
        finding.model_dump(mode="json") for finding in snapshot.overlap_findings
    ]

    record.document_warnings = list(snapshot.document_warnings)

    record.review_result = (
        snapshot.review_result.model_dump(mode="json")
        if snapshot.review_result is not None
        else None
    )


def insert_approved_evidence(
    *,
    session: Session,
    user_id: str,
    evidence: CareerEvidence,
) -> None:
    """Insert one newly approved evidence record."""

    if evidence.verification_status is not VerificationStatus.APPROVED:
        raise ValueError(
            "CV evidence persistence accepts only approved career evidence."
        )

    existing = session.get(
        CareerEvidenceRecord,
        (
            user_id,
            evidence.evidence_id,
        ),
    )

    if existing is not None:
        raise ValueError("Approved evidence identifier already exists.")

    session.add(
        CareerEvidenceRecord(
            user_id=user_id,
            evidence_id=evidence.evidence_id,
            category=evidence.category.value,
            title=evidence.title,
            verification_status=(evidence.verification_status.value),
            technologies=list(evidence.technologies),
            capabilities=list(evidence.capabilities),
            approved_claims=list(evidence.approved_claims),
            source_references=[
                reference.model_dump(mode="json")
                for reference in evidence.source_references
            ],
        )
    )


def review_run_record_to_domain(
    record: CVEvidenceReviewRunRecord,
) -> CVEvidenceReviewRunSnapshot:
    """Convert a persistent run snapshot to its domain model."""

    return CVEvidenceReviewRunSnapshot(
        review_run_id=record.review_run_id,
        user_id=record.user_id,
        document_id=record.document_id,
        status=CVEvidenceReviewRunStatus(record.status),
        proposals=[
            CareerEvidenceProposal.model_validate(proposal)
            for proposal in record.proposals
        ],
        overlap_findings=[
            CareerEvidenceOverlapFinding.model_validate(finding)
            for finding in record.overlap_findings
        ],
        document_warnings=list(record.document_warnings),
        review_result=(
            EvidenceReviewResult.model_validate(record.review_result)
            if record.review_result is not None
            else None
        ),
    )


def review_run_record_to_summary(
    record: CVEvidenceReviewRunRecord,
) -> CVEvidenceReviewRunSummary:
    """Convert a persisted review run to a lightweight history summary."""

    review_result = (
        EvidenceReviewResult.model_validate(record.review_result)
        if record.review_result is not None
        else None
    )

    return CVEvidenceReviewRunSummary(
        review_run_id=record.review_run_id,
        document_id=record.document_id,
        status=CVEvidenceReviewRunStatus(record.status),
        proposal_count=len(record.proposals),
        approved_evidence_count=(
            len(review_result.approved_evidence) if review_result is not None else 0
        ),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def review_history_record_to_domain(
    record: CVEvidenceReviewHistoryRecord,
) -> CVEvidenceReviewAuditEntry:
    """Convert one persistent audit row to its domain model."""

    return CVEvidenceReviewAuditEntry(
        review_id=record.review_id,
        review_run_id=record.review_run_id,
        sequence_number=record.sequence_number,
        decision=EvidenceReviewDecision.model_validate(record.decision_payload),
        result=EvidenceReviewResult.model_validate(record.result_payload),
        recorded_at=record.created_at,
    )


def validate_idempotent_review_retry(
    *,
    session: Session,
    snapshot: CVEvidenceReviewRunSnapshot,
    review: CVEvidenceReviewAuditEntry,
    existing_review: CVEvidenceReviewHistoryRecord,
) -> None:
    """Accept an exact retry while rejecting conflicting reuse."""

    stored_review = review_history_record_to_domain(existing_review)

    if not reviews_are_equivalent(
        stored_review,
        review,
    ):
        raise ValueError("Review identifier already exists with different audit data.")

    run_record = session.get(
        CVEvidenceReviewRunRecord,
        snapshot.review_run_id,
    )

    if run_record is None or review_run_record_to_domain(run_record) != snapshot:
        raise ValueError("Retry snapshot does not match the persisted CV evidence run.")

    for evidence in review.result.approved_evidence:
        record = session.get(
            CareerEvidenceRecord,
            (
                snapshot.user_id,
                evidence.evidence_id,
            ),
        )

        if record is None or record_to_domain(record) != evidence:
            raise ValueError(
                "Retry evidence does not match the persisted approved evidence."
            )


def reviews_are_equivalent(
    first: CVEvidenceReviewAuditEntry,
    second: CVEvidenceReviewAuditEntry,
) -> bool:
    """Compare retry-relevant data while ignoring DB timestamp."""

    return first.model_dump(
        mode="json",
        exclude={"recorded_at"},
    ) == second.model_dump(
        mode="json",
        exclude={"recorded_at"},
    )
