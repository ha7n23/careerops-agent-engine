"""SQLAlchemy repository for persistent approved career evidence."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceLifecycleStatus,
    EvidenceMutationAction,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceEdit,
    SourceReference,
)
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceHistoryRecord,
    CareerEvidenceRecord,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    build_searchable_text,
    tokenise,
)

MAX_SEARCH_CANDIDATES = 500


class SqlAlchemyEvidenceRepository:
    """Retrieve approved evidence from the CareerOps database."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Store the configured SQLAlchemy session factory."""

        self._session_factory = session_factory

    def search_approved(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[CareerEvidence]:
        """Search approved user evidence using deterministic token overlap."""

        if limit < 1:
            raise ValueError("Search limit must be at least one.")

        query_tokens = tokenise(query)

        if not query_tokens:
            return []

        candidates = self._load_approved_candidates(
            user_id=user_id,
            limit=MAX_SEARCH_CANDIDATES,
        )

        scored_evidence: list[tuple[int, CareerEvidence]] = []

        for evidence in candidates:
            evidence_tokens = tokenise(build_searchable_text(evidence))
            score = len(query_tokens & evidence_tokens)

            if score > 0:
                scored_evidence.append((score, evidence))

        scored_evidence.sort(
            key=lambda item: (
                -item[0],
                item[1].title.casefold(),
                item[1].evidence_id,
            )
        )

        return [evidence for _, evidence in scored_evidence[:limit]]

    def get_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence | None:
        """Retrieve one approved record within the user boundary."""

        statement = select(CareerEvidenceRecord).where(
            CareerEvidenceRecord.user_id == user_id,
            CareerEvidenceRecord.evidence_id == evidence_id,
            CareerEvidenceRecord.verification_status
            == VerificationStatus.APPROVED.value,
            CareerEvidenceRecord.lifecycle_status
            == EvidenceLifecycleStatus.ACTIVE.value,
        )

        with self._session_factory() as session:
            record = session.execute(statement).scalar_one_or_none()

        if record is None:
            return None

        return record_to_domain(record)

    def list_approved(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[CareerEvidence]:
        """List a bounded set of approved user evidence."""

        if limit < 1:
            raise ValueError("List limit must be at least one.")

        return self._load_approved_candidates(
            user_id=user_id,
            limit=limit,
        )

    def get_approved_for_management(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence | None:
        """Retrieve active or archived approved evidence for its owner."""

        statement = select(CareerEvidenceRecord).where(
            CareerEvidenceRecord.user_id == user_id,
            CareerEvidenceRecord.evidence_id == evidence_id,
            CareerEvidenceRecord.verification_status
            == VerificationStatus.APPROVED.value,
        )

        with self._session_factory() as session:
            record = session.execute(statement).scalar_one_or_none()

        return record_to_domain(record) if record is not None else None

    def edit_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
        edit: CareerEvidenceEdit,
    ) -> CareerEvidence | None:
        """Apply and audit a strict edit under a row lock."""

        with self._session_factory.begin() as session:
            record = load_managed_record_for_update(
                session=session,
                user_id=user_id,
                evidence_id=evidence_id,
            )

            if record is None:
                return None

            before = record_to_domain(record)
            changes = edit.model_dump(
                mode="json",
                exclude_none=True,
            )

            for field_name, value in changes.items():
                setattr(record, field_name, value)

            after = record_to_domain(record)

            if after == before:
                return before

            add_history_entry(
                session=session,
                user_id=user_id,
                evidence_id=evidence_id,
                action=EvidenceMutationAction.EDIT,
                before=before,
                after=after,
            )

            return after

    def set_lifecycle_status(
        self,
        *,
        user_id: str,
        evidence_id: str,
        lifecycle_status: EvidenceLifecycleStatus,
    ) -> CareerEvidence | None:
        """Idempotently archive or restore evidence under a row lock."""

        with self._session_factory.begin() as session:
            record = load_managed_record_for_update(
                session=session,
                user_id=user_id,
                evidence_id=evidence_id,
            )

            if record is None:
                return None

            before = record_to_domain(record)

            if before.lifecycle_status is lifecycle_status:
                return before

            record.lifecycle_status = lifecycle_status.value
            record.archived_at = (
                datetime.now(UTC)
                if lifecycle_status is EvidenceLifecycleStatus.ARCHIVED
                else None
            )

            after = record_to_domain(record)
            action = (
                EvidenceMutationAction.ARCHIVE
                if lifecycle_status is EvidenceLifecycleStatus.ARCHIVED
                else EvidenceMutationAction.RESTORE
            )

            add_history_entry(
                session=session,
                user_id=user_id,
                evidence_id=evidence_id,
                action=action,
                before=before,
                after=after,
            )

            return after

    def _load_approved_candidates(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[CareerEvidence]:
        """Load ordered approved evidence for one user."""

        statement = (
            select(CareerEvidenceRecord)
            .where(
                CareerEvidenceRecord.user_id == user_id,
                CareerEvidenceRecord.verification_status
                == VerificationStatus.APPROVED.value,
                CareerEvidenceRecord.lifecycle_status
                == EvidenceLifecycleStatus.ACTIVE.value,
            )
            .order_by(
                CareerEvidenceRecord.title,
                CareerEvidenceRecord.evidence_id,
            )
            .limit(limit)
        )

        with self._session_factory() as session:
            records = session.execute(statement).scalars().all()

        return [record_to_domain(record) for record in records]


def record_to_domain(
    record: CareerEvidenceRecord,
) -> CareerEvidence:
    """Convert a persistent evidence record into a domain model."""

    return CareerEvidence(
        evidence_id=record.evidence_id,
        category=EvidenceCategory(record.category),
        title=record.title,
        verification_status=VerificationStatus(record.verification_status),
        lifecycle_status=EvidenceLifecycleStatus(record.lifecycle_status),
        technologies=list(record.technologies),
        capabilities=list(record.capabilities),
        approved_claims=list(record.approved_claims),
        source_references=[
            SourceReference.model_validate(reference)
            for reference in record.source_references
        ],
    )


def load_managed_record_for_update(
    *,
    session: Session,
    user_id: str,
    evidence_id: str,
) -> CareerEvidenceRecord | None:
    """Load one approved user-owned record and lock it for mutation."""

    statement = (
        select(CareerEvidenceRecord)
        .where(
            CareerEvidenceRecord.user_id == user_id,
            CareerEvidenceRecord.evidence_id == evidence_id,
            CareerEvidenceRecord.verification_status
            == VerificationStatus.APPROVED.value,
        )
        .with_for_update()
    )

    return session.execute(statement).scalar_one_or_none()


def add_history_entry(
    *,
    session: Session,
    user_id: str,
    evidence_id: str,
    action: EvidenceMutationAction,
    before: CareerEvidence,
    after: CareerEvidence,
) -> None:
    """Append one immutable registry mutation record."""

    session.add(
        CareerEvidenceHistoryRecord(
            event_id=f"EVH-{uuid4().hex.upper()}",
            user_id=user_id,
            evidence_id=evidence_id,
            action=action.value,
            before_payload=before.model_dump(mode="json"),
            after_payload=after.model_dump(mode="json"),
        )
    )
