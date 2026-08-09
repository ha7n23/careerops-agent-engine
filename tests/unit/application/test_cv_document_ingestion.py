"""Tests for persistent CV upload orchestration."""

from careerops_agent_engine.application.services.cv_document_ingestion import (
    CVDocumentIngestionService,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
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
