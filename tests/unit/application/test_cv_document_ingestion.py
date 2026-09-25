"""Tests for persistent CV upload orchestration."""

from hashlib import sha256

import pytest

from careerops_agent_engine.application.exceptions import (
    DocumentUploadValidationError,
)
from careerops_agent_engine.application.services.cv_document_ingestion import (
    CVDocumentIngestionService,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import CareerDocument


class FakeStorage:
    """Keep document bytes in memory."""

    def __init__(self) -> None:
        self.saved: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        key = f"documents/{user_id}/{document_id}.{document_format.value}"

        self.saved[key] = data

        return key

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        del user_id
        return self.saved[storage_key]

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        del user_id
        self.deleted.append(storage_key)
        self.saved.pop(storage_key, None)


class FakeDocumentRepository:
    """Capture persisted document metadata."""

    def __init__(
        self,
        *,
        fail_save: bool = False,
    ) -> None:
        self.fail_save = fail_save
        self.saved: list[tuple[str, CareerDocument]] = []

    def save(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> None:
        if self.fail_save:
            raise RuntimeError("Database unavailable")

        self.saved.append(
            (
                user_id,
                document,
            )
        )

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CareerDocument | None:
        """Return a matching stored document when present."""

        for saved_user_id, document in self.saved:
            if saved_user_id == user_id and document.document_id == document_id:
                return document

        return None


def test_ingest_persists_document_metadata() -> None:
    """Validated file storage should be followed by metadata persistence."""

    storage = FakeStorage()
    repository = FakeDocumentRepository()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=repository,
        storage=storage,
    )

    document = service.ingest(
        user_id="USER-001",
        original_filename="cv.pdf",
        declared_media_type="application/pdf",
        data=b"%PDF-1.7\nCV bytes",
    )

    assert len(repository.saved) == 1

    saved_user_id, saved_document = repository.saved[0]

    assert saved_user_id == "USER-001"
    assert saved_document == document

    assert list(storage.saved.values()) == [b"%PDF-1.7\nCV bytes"]


def test_metadata_failure_cleans_up_stored_bytes() -> None:
    """A failed metadata transaction must not leave an orphaned file."""

    storage = FakeStorage()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=FakeDocumentRepository(fail_save=True),
        storage=storage,
    )

    try:
        service.ingest(
            user_id="USER-001",
            original_filename="cv.pdf",
            declared_media_type="application/pdf",
            data=b"%PDF-1.7\nCV bytes",
        )
    except RuntimeError as exc:
        assert str(exc) == ("Database unavailable")
    else:
        raise AssertionError("Expected metadata persistence failure.")

    assert storage.saved == {}
    assert len(storage.deleted) == 1


def test_ingest_text_persists_normalised_utf8_source() -> None:
    """Pasted evidence should use the trusted text-document pipeline."""

    storage = FakeStorage()
    repository = FakeDocumentRepository()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=repository,
        storage=storage,
        max_text_characters=100,
    )

    document = service.ingest_text(
        user_id="USER-001",
        title="  AWS deployment notes  ",
        content=(
            "  Deployed a FastAPI application to AWS.\r\n"
            "Containerised the service using Docker.  "
        ),
    )

    expected_data = (
        b"Deployed a FastAPI application to AWS.\n"
        b"Containerised the service using Docker."
    )

    assert document.original_filename == "AWS deployment notes.txt"
    assert document.document_format is CareerDocumentFormat.TEXT
    assert document.media_type == "text/plain; charset=utf-8"
    assert document.status is CareerDocumentStatus.UPLOADED
    assert document.size_bytes == len(expected_data)
    assert document.sha256_hex == sha256(expected_data).hexdigest()

    assert storage.saved[document.storage_key] == expected_data
    assert repository.saved == [("USER-001", document)]


def test_ingest_text_rejects_blank_content_before_storage() -> None:
    """Whitespace-only textarea content must not create a document."""

    storage = FakeStorage()
    repository = FakeDocumentRepository()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=repository,
        storage=storage,
    )

    with pytest.raises(
        DocumentUploadValidationError,
        match="cannot be empty",
    ):
        service.ingest_text(
            user_id="USER-001",
            title="Notes",
            content=" \r\n\t ",
        )

    assert storage.saved == {}
    assert repository.saved == []


def test_ingest_text_rejects_content_above_character_limit() -> None:
    """Oversized pasted evidence must stop before storage."""

    storage = FakeStorage()
    repository = FakeDocumentRepository()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=repository,
        storage=storage,
        max_text_characters=10,
    )

    with pytest.raises(
        DocumentUploadValidationError,
        match="maximum allowed length",
    ):
        service.ingest_text(
            user_id="USER-001",
            title="Notes",
            content="12345678901",
        )

    assert storage.saved == {}
    assert repository.saved == []


def test_ingest_text_rejects_blank_title() -> None:
    """Every pasted source requires a frontend-visible title."""

    storage = FakeStorage()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=FakeDocumentRepository(),
        storage=storage,
    )

    with pytest.raises(
        DocumentUploadValidationError,
        match="requires a title",
    ):
        service.ingest_text(
            user_id="USER-001",
            title=" \t ",
            content="Built an API using FastAPI.",
        )

    assert storage.saved == {}


def test_text_metadata_failure_removes_stored_bytes() -> None:
    """A database failure must not leave orphaned pasted-text bytes."""

    storage = FakeStorage()

    service = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=1024,
        ),
        repository=FakeDocumentRepository(fail_save=True),
        storage=storage,
    )

    with pytest.raises(
        RuntimeError,
        match="Database unavailable",
    ):
        service.ingest_text(
            user_id="USER-001",
            title="Python evidence",
            content="Built Python APIs using FastAPI.",
        )

    assert storage.saved == {}
    assert len(storage.deleted) == 1


def test_ingestion_requires_positive_text_limit() -> None:
    """The pasted-text safety boundary cannot be disabled."""

    storage = FakeStorage()

    with pytest.raises(
        ValueError,
        match="text character count must be positive",
    ):
        CVDocumentIngestionService(
            upload_service=CVDocumentUploadService(
                storage=storage,
                max_upload_bytes=1024,
            ),
            repository=FakeDocumentRepository(),
            storage=storage,
            max_text_characters=0,
        )
