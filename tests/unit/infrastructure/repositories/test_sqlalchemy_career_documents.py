"""Tests for persistent career-document metadata."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_career_documents import (  # noqa: E501
    SqlAlchemyCareerDocumentRepository,
)


def build_document(
    *,
    status: CareerDocumentStatus = (CareerDocumentStatus.UPLOADED),
    filename: str = "cv.pdf",
) -> CareerDocument:
    """Create trusted document metadata."""

    return CareerDocument(
        document_id="DOC-001",
        original_filename=filename,
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key=("documents/usr-test/DOC-001.pdf"),
        status=status,
    )


@pytest.fixture
def repository() -> Iterator[SqlAlchemyCareerDocumentRepository]:
    """Create an isolated metadata repository."""

    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")

    Base.metadata.create_all(engine)

    try:
        yield SqlAlchemyCareerDocumentRepository(create_session_factory(engine))
    finally:
        engine.dispose()


def test_save_and_get_document_round_trip(
    repository: SqlAlchemyCareerDocumentRepository,
) -> None:
    """Stored metadata should round-trip exactly."""

    document = build_document()

    repository.save(
        user_id="USER-001",
        document=document,
    )

    assert (
        repository.get(
            user_id="USER-001",
            document_id="DOC-001",
        )
        == document
    )


def test_document_get_enforces_user_boundary(
    repository: SqlAlchemyCareerDocumentRepository,
) -> None:
    """Another user cannot retrieve document metadata."""

    repository.save(
        user_id="USER-001",
        document=build_document(),
    )

    assert (
        repository.get(
            user_id="USER-OTHER",
            document_id="DOC-001",
        )
        is None
    )


def test_document_status_may_be_updated(
    repository: SqlAlchemyCareerDocumentRepository,
) -> None:
    """Lifecycle status may change without rewriting identity."""

    repository.save(
        user_id="USER-001",
        document=build_document(),
    )

    extracted = build_document(status=CareerDocumentStatus.EXTRACTED)

    repository.save(
        user_id="USER-001",
        document=extracted,
    )

    assert (
        repository.get(
            user_id="USER-001",
            document_id="DOC-001",
        )
        == extracted
    )


def test_document_identity_metadata_cannot_change(
    repository: SqlAlchemyCareerDocumentRepository,
) -> None:
    """An existing stored file identity must remain immutable."""

    repository.save(
        user_id="USER-001",
        document=build_document(),
    )

    with pytest.raises(
        ValueError,
        match="cannot change its stored identity metadata",
    ):
        repository.save(
            user_id="USER-001",
            document=build_document(filename="different.pdf"),
        )


def test_document_summaries_are_bounded_newest_first_and_user_scoped(
    repository: SqlAlchemyCareerDocumentRepository,
) -> None:
    """Document history should be bounded and remain inside its user scope."""

    first_document = build_document()

    second_document = build_document().model_copy(
        update={
            "document_id": "DOC-002",
            "original_filename": "newer-cv.pdf",
            "storage_key": "documents/usr-test/DOC-002.pdf",
        }
    )

    repository.save(
        user_id="USER-001",
        document=first_document,
    )
    repository.save(
        user_id="USER-001",
        document=second_document,
    )

    summaries = repository.list_summaries(
        user_id="USER-001",
        limit=1,
    )

    assert len(summaries) == 1
    assert summaries[0].document_id == "DOC-002"
    assert summaries[0].original_filename == "newer-cv.pdf"
    assert summaries[0].uploaded_at is not None
    assert summaries[0].updated_at is not None

    assert (
        repository.list_summaries(
            user_id="USER-OTHER",
            limit=10,
        )
        == []
    )
