"""SQLAlchemy repository for persistent approved career evidence."""

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)
from careerops_agent_engine.infrastructure.database.models.evidence import (
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
        technologies=list(record.technologies),
        capabilities=list(record.capabilities),
        approved_claims=list(record.approved_claims),
        source_references=[
            SourceReference.model_validate(reference)
            for reference in record.source_references
        ],
    )
