"""Tests for the CareerOps CV-document upload API."""

from collections.abc import Iterator
from io import BytesIO
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_cv_document_ingestion_service,
)
from careerops_agent_engine.application.services.cv_document_ingestion import (
    CVDocumentIngestionService,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.main import app


class FakeDocumentStorage:
    """Keep uploaded bytes in memory for API tests."""

    def __init__(self) -> None:
        """Create empty document storage."""

        self.saved: dict[str, bytes] = {}

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Store bytes under a deterministic test key."""

        key = f"documents/{user_id}/{document_id}.{document_format.value}"

        self.saved[key] = data

        return key

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Read stored test bytes."""

        del user_id

        return self.saved[storage_key]

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Delete stored test bytes."""

        del user_id

        del self.saved[storage_key]


class FakeDocumentRepository:
    """Persist uploaded document metadata in memory."""

    def __init__(self) -> None:
        self.documents: dict[
            tuple[str, str],
            CareerDocument,
        ] = {}

    def save(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> None:
        """Persist the latest document metadata."""

        self.documents[
            (
                user_id,
                document.document_id,
            )
        ] = document

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CareerDocument | None:
        """Retrieve metadata within its user boundary."""

        return self.documents.get(
            (
                user_id,
                document_id,
            )
        )


def build_docx_bytes() -> bytes:
    """Create a minimal DOCX-shaped test archive."""

    output = BytesIO()

    with ZipFile(
        output,
        mode="w",
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types />",
        )
        archive.writestr(
            "word/document.xml",
            "<document />",
        )

    return output.getvalue()


@pytest.fixture
def document_storage() -> FakeDocumentStorage:
    """Provide isolated uploaded-byte storage."""

    return FakeDocumentStorage()


@pytest.fixture
def document_repository() -> FakeDocumentRepository:
    """Provide isolated document metadata persistence."""

    return FakeDocumentRepository()


@pytest.fixture
def client(
    document_storage: FakeDocumentStorage,
    document_repository: FakeDocumentRepository,
) -> Iterator[TestClient]:
    """Create an API client with fake document storage."""

    upload_service = CVDocumentUploadService(
        storage=document_storage,
        max_upload_bytes=1024,
    )

    service = CVDocumentIngestionService(
        upload_service=upload_service,
        repository=document_repository,
        storage=document_storage,
        max_text_characters=100,
    )

    app.dependency_overrides[get_cv_document_ingestion_service] = lambda: service

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(
            get_cv_document_ingestion_service,
            None,
        )


def test_pdf_upload_returns_safe_metadata(
    client: TestClient,
    document_storage: FakeDocumentStorage,
    document_repository: FakeDocumentRepository,
) -> None:
    """A valid PDF should return 201 without storage internals."""

    data = b"%PDF-1.7\nCareerOps test CV"

    response = client.post(
        "/api/v1/cv-documents",
        headers={
            "X-User-ID": "USER-001",
        },
        files={
            "file": (
                "cv.pdf",
                data,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    body = response.json()

    assert body["original_filename"] == "cv.pdf"

    assert body["document_format"] == "pdf"

    assert body["media_type"] == "application/pdf"

    assert body["size_bytes"] == len(data)

    assert body["status"] == "uploaded"

    assert len(body["sha256_hex"]) == 64

    assert body["document_id"].startswith("DOC-")

    # Infrastructure location must remain private.
    assert "storage_key" not in body

    assert list(document_storage.saved.values()) == [data]

    persisted = document_repository.get(
        user_id="USER-001",
        document_id=body["document_id"],
    )

    assert persisted is not None
    assert persisted.original_filename == "cv.pdf"
    assert persisted.sha256_hex == body["sha256_hex"]


def test_docx_upload_returns_created(
    client: TestClient,
) -> None:
    """A valid DOCX multipart upload should be accepted."""

    data = build_docx_bytes()

    response = client.post(
        "/api/v1/cv-documents",
        headers={
            "X-User-ID": "USER-001",
        },
        files={
            "file": (
                "cv.docx",
                data,
                (
                    "application/vnd."
                    "openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["document_format"] == "docx"


def test_invalid_document_returns_422(
    client: TestClient,
) -> None:
    """Arbitrary bytes must not cross the HTTP upload boundary."""

    response = client.post(
        "/api/v1/cv-documents",
        headers={
            "X-User-ID": "USER-001",
        },
        files={
            "file": (
                "cv.pdf",
                b"not a real document",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422

    assert "Only valid PDF and DOCX" in response.json()["detail"]


def test_mismatched_extension_returns_422(
    client: TestClient,
) -> None:
    """A fake extension must not override detected bytes."""

    response = client.post(
        "/api/v1/cv-documents",
        headers={
            "X-User-ID": "USER-001",
        },
        files={
            "file": (
                "cv.docx",
                b"%PDF-1.7\nPDF bytes",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 422

    assert "extension does not match" in response.json()["detail"]


def test_oversized_upload_returns_422(
    client: TestClient,
    document_storage: FakeDocumentStorage,
) -> None:
    """The HTTP boundary must enforce the configured size limit."""

    data = b"%PDF-1.7\n" + b"x" * 2_000

    response = client.post(
        "/api/v1/cv-documents",
        headers={
            "X-User-ID": "USER-001",
        },
        files={
            "file": (
                "cv.pdf",
                data,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422

    assert "maximum allowed size" in response.json()["detail"]

    assert document_storage.saved == {}


def test_missing_user_header_returns_401(
    client: TestClient,
) -> None:
    """Uploading a CV requires authenticated user context."""

    response = client.post(
        "/api/v1/cv-documents",
        files={
            "file": (
                "cv.pdf",
                b"%PDF-1.7\nPDF bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 401


def test_user_path_in_filename_is_not_returned(
    client: TestClient,
) -> None:
    """Client filesystem paths must be stripped from metadata."""

    response = client.post(
        "/api/v1/cv-documents",
        headers={
            "X-User-ID": "USER-001",
        },
        files={
            "file": (
                r"C:\Users\candidate\cv.pdf",
                b"%PDF-1.7\nPDF bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    assert response.json()["original_filename"] == "cv.pdf"


def test_text_evidence_submission_returns_safe_metadata(
    client: TestClient,
    document_storage: FakeDocumentStorage,
    document_repository: FakeDocumentRepository,
) -> None:
    """Valid textarea evidence should enter the shared document pipeline."""

    response = client.post(
        "/api/v1/cv-documents/text",
        headers={
            "X-User-ID": "USER-001",
        },
        json={
            "title": "AWS deployment evidence",
            "content": (
                "Deployed a containerised FastAPI application to AWS.\r\n"
                "Used Docker for packaging."
            ),
        },
    )

    assert response.status_code == 201

    body = response.json()

    expected_data = (
        b"Deployed a containerised FastAPI application to AWS.\n"
        b"Used Docker for packaging."
    )

    assert body["original_filename"] == "AWS deployment evidence.txt"
    assert body["document_format"] == "text"
    assert body["media_type"] == "text/plain; charset=utf-8"
    assert body["size_bytes"] == len(expected_data)
    assert body["status"] == "uploaded"
    assert body["document_id"].startswith("DOC-")
    assert "storage_key" not in body

    assert list(document_storage.saved.values()) == [expected_data]

    persisted = document_repository.get(
        user_id="USER-001",
        document_id=body["document_id"],
    )

    assert persisted is not None
    assert persisted.document_format is CareerDocumentFormat.TEXT


def test_blank_text_evidence_returns_422(
    client: TestClient,
    document_storage: FakeDocumentStorage,
) -> None:
    """Whitespace-only textarea content must fail before storage."""

    response = client.post(
        "/api/v1/cv-documents/text",
        headers={
            "X-User-ID": "USER-001",
        },
        json={
            "title": "Empty notes",
            "content": " \r\n\t ",
        },
    )

    assert response.status_code == 422
    assert "cannot be empty" in response.json()["detail"]
    assert document_storage.saved == {}


def test_text_evidence_above_runtime_limit_returns_422(
    client: TestClient,
    document_storage: FakeDocumentStorage,
) -> None:
    """The configured pasted-text boundary must be enforced."""

    response = client.post(
        "/api/v1/cv-documents/text",
        headers={
            "X-User-ID": "USER-001",
        },
        json={
            "title": "Oversized notes",
            "content": "A" * 101,
        },
    )

    assert response.status_code == 422
    assert "maximum allowed length" in response.json()["detail"]
    assert document_storage.saved == {}


def test_text_evidence_requires_authenticated_user(
    client: TestClient,
) -> None:
    """Textarea ingestion must retain the existing user boundary."""

    response = client.post(
        "/api/v1/cv-documents/text",
        json={
            "title": "Python evidence",
            "content": "Built Python APIs using FastAPI.",
        },
    )

    assert response.status_code == 401


def test_text_evidence_rejects_client_supplied_user_id(
    client: TestClient,
) -> None:
    """The request body must not allow user-scope impersonation."""

    response = client.post(
        "/api/v1/cv-documents/text",
        headers={
            "X-User-ID": "USER-001",
        },
        json={
            "title": "Python evidence",
            "content": "Built Python APIs using FastAPI.",
            "user_id": "USER-OTHER",
        },
    )

    assert response.status_code == 422
