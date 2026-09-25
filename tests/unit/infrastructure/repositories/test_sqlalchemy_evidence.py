"""Tests for the SQLAlchemy evidence repository."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceLifecycleStatus,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import CareerEvidenceEdit
from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceHistoryRecord,
    CareerEvidenceRecord,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)


def build_record(
    *,
    user_id: str,
    evidence_id: str,
    title: str,
    category: EvidenceCategory,
    status: VerificationStatus,
    technologies: list[str],
    claims: list[str],
) -> CareerEvidenceRecord:
    """Create one persistent evidence record for tests."""

    return CareerEvidenceRecord(
        user_id=user_id,
        evidence_id=evidence_id,
        title=title,
        category=category.value,
        verification_status=status.value,
        technologies=technologies,
        capabilities=[],
        approved_claims=claims,
        source_references=[
            {
                "source_type": "manual_entry",
                "source_id": f"SRC-{evidence_id}",
                "page_number": None,
                "source_excerpt": None,
            }
        ],
    )


@pytest.fixture
def repository() -> Iterator[SqlAlchemyEvidenceRepository]:
    """Create an isolated SQLite-backed repository."""

    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add_all(
            [
                build_record(
                    user_id="USER-001",
                    evidence_id="EVD-DOCKER",
                    title="Containerised AI Service",
                    category=EvidenceCategory.PROJECT,
                    status=VerificationStatus.APPROVED,
                    technologies=["Python", "Docker"],
                    claims=["Containerised a FastAPI service using Docker."],
                ),
                build_record(
                    user_id="USER-001",
                    evidence_id="EVD-REJECTED-K8S",
                    title="Unverified Kubernetes Claim",
                    category=EvidenceCategory.SKILL,
                    status=VerificationStatus.REJECTED,
                    technologies=["Kubernetes"],
                    claims=["Used Kubernetes in production."],
                ),
                build_record(
                    user_id="USER-002",
                    evidence_id="EVD-OTHER-K8S",
                    title="Kubernetes Deployment",
                    category=EvidenceCategory.PROJECT,
                    status=VerificationStatus.APPROVED,
                    technologies=["Kubernetes"],
                    claims=["Deployed an application to Kubernetes."],
                ),
            ]
        )
        session.commit()

    try:
        yield SqlAlchemyEvidenceRepository(session_factory)
    finally:
        engine.dispose()


def test_search_returns_approved_matching_evidence(
    repository: SqlAlchemyEvidenceRepository,
) -> None:
    """Matching approved evidence should be returned."""

    results = repository.search_approved(
        user_id="USER-001",
        query="Docker containerised service",
    )

    assert [result.evidence_id for result in results] == ["EVD-DOCKER"]


def test_search_excludes_rejected_and_cross_user_records(
    repository: SqlAlchemyEvidenceRepository,
) -> None:
    """Search must enforce approval and user isolation."""

    results = repository.search_approved(
        user_id="USER-001",
        query="Kubernetes",
    )

    assert results == []


def test_get_approved_enforces_user_boundary(
    repository: SqlAlchemyEvidenceRepository,
) -> None:
    """Another user's approved evidence must remain inaccessible."""

    result = repository.get_approved(
        user_id="USER-001",
        evidence_id="EVD-OTHER-K8S",
    )

    assert result is None


def test_list_returns_only_approved_user_evidence(
    repository: SqlAlchemyEvidenceRepository,
) -> None:
    """Listing should expose only approved records for one user."""

    results = repository.list_approved(
        user_id="USER-001",
    )

    assert [result.evidence_id for result in results] == ["EVD-DOCKER"]


def test_archive_restore_and_edit_are_audited_and_idempotent(
    repository: SqlAlchemyEvidenceRepository,
) -> None:
    """Each meaningful mutation gets one audit entry; retries do not."""

    archived = repository.set_lifecycle_status(
        user_id="USER-001",
        evidence_id="EVD-DOCKER",
        lifecycle_status=EvidenceLifecycleStatus.ARCHIVED,
    )
    retry = repository.set_lifecycle_status(
        user_id="USER-001",
        evidence_id="EVD-DOCKER",
        lifecycle_status=EvidenceLifecycleStatus.ARCHIVED,
    )

    assert archived is not None
    assert retry == archived
    assert repository.list_approved(user_id="USER-001") == []
    assert (
        repository.get_approved(
            user_id="USER-001",
            evidence_id="EVD-DOCKER",
        )
        is None
    )
    assert (
        repository.get_approved_for_management(
            user_id="USER-001",
            evidence_id="EVD-DOCKER",
        )
        == archived
    )

    restored = repository.set_lifecycle_status(
        user_id="USER-001",
        evidence_id="EVD-DOCKER",
        lifecycle_status=EvidenceLifecycleStatus.ACTIVE,
    )
    edited = repository.edit_approved(
        user_id="USER-001",
        evidence_id="EVD-DOCKER",
        edit=CareerEvidenceEdit(title="Containerised CareerOps Service"),
    )

    assert restored is not None
    assert edited is not None
    assert edited.title == "Containerised CareerOps Service"

    with repository._session_factory() as session:
        history = session.query(CareerEvidenceHistoryRecord).all()
        actions = [record.action for record in history]

    assert len(actions) == 3
    assert set(actions) == {"archive", "restore", "edit"}


def test_mutations_keep_unknown_and_cross_user_ids_opaque(
    repository: SqlAlchemyEvidenceRepository,
) -> None:
    """Mutation methods must enforce the same ownership boundary as reads."""

    for evidence_id in ("EVD-OTHER-K8S", "EVD-MISSING"):
        assert (
            repository.edit_approved(
                user_id="USER-001",
                evidence_id=evidence_id,
                edit=CareerEvidenceEdit(title="Unavailable"),
            )
            is None
        )
        assert (
            repository.set_lifecycle_status(
                user_id="USER-001",
                evidence_id=evidence_id,
                lifecycle_status=EvidenceLifecycleStatus.ARCHIVED,
            )
            is None
        )
