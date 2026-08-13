"""Tests for private local career-document storage."""

from pathlib import Path

import pytest

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)
from careerops_agent_engine.infrastructure.storage import (
    local_document_storage,
)


def build_storage(
    tmp_path: Path,
) -> local_document_storage.LocalDocumentStorage:
    """Create isolated private storage."""

    return (
        local_document_storage.LocalDocumentStorage(
            tmp_path / "careerops-documents"
        )
    )


def test_save_and_read_round_trip(
    tmp_path: Path,
) -> None:
    """Validated bytes should round-trip through local storage."""

    storage = build_storage(
        tmp_path
    )

    data = b"%PDF-1.7\nCareerOps CV"

    storage_key = storage.save(
        user_id="USER-001",
        document_id="DOC-001",
        document_format=CareerDocumentFormat.PDF,
        data=data,
    )

    loaded = storage.read(
        user_id="USER-001",
        storage_key=storage_key,
    )

    assert loaded == data


def test_storage_key_does_not_expose_raw_user_id(
    tmp_path: Path,
) -> None:
    """Storage paths should use an opaque user namespace."""

    storage = build_storage(
        tmp_path
    )

    storage_key = storage.save(
        user_id="USER-SENSITIVE-001",
        document_id="DOC-001",
        document_format=CareerDocumentFormat.PDF,
        data=b"%PDF-1.7\nCV",
    )

    assert "USER-SENSITIVE-001" not in storage_key
    assert storage_key.startswith(
        "documents/usr-"
    )


def test_another_user_cannot_read_document(
    tmp_path: Path,
) -> None:
    """A storage key must remain bound to its owner."""

    storage = build_storage(
        tmp_path
    )

    storage_key = storage.save(
        user_id="USER-001",
        document_id="DOC-001",
        document_format=CareerDocumentFormat.PDF,
        data=b"%PDF-1.7\nCV",
    )

    with pytest.raises(
        FileNotFoundError,
        match="unavailable",
    ):
        storage.read(
            user_id="USER-OTHER",
            storage_key=storage_key,
        )


def test_path_traversal_storage_key_is_rejected(
    tmp_path: Path,
) -> None:
    """A supplied key cannot escape the private root."""

    storage = build_storage(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="unavailable",
    ):
        storage.read(
            user_id="USER-001",
            storage_key="../../secret.pdf",
        )


def test_unsafe_document_id_is_rejected(
    tmp_path: Path,
) -> None:
    """Document identifiers cannot control filesystem paths."""

    storage = build_storage(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="unsafe characters",
    ):
        storage.save(
            user_id="USER-001",
            document_id="../../DOC-001",
            document_format=CareerDocumentFormat.PDF,
            data=b"%PDF-1.7\nCV",
        )


def test_delete_removes_document(
    tmp_path: Path,
) -> None:
    """Deleting a document should make its bytes unavailable."""

    storage = build_storage(
        tmp_path
    )

    storage_key = storage.save(
        user_id="USER-001",
        document_id="DOC-001",
        document_format=CareerDocumentFormat.PDF,
        data=b"%PDF-1.7\nCV",
    )

    storage.delete(
        user_id="USER-001",
        storage_key=storage_key,
    )

    with pytest.raises(
        FileNotFoundError,
        match="unavailable",
    ):
        storage.read(
            user_id="USER-001",
            storage_key=storage_key,
        )


def test_duplicate_save_does_not_overwrite_existing_bytes(
    tmp_path: Path,
) -> None:
    """An identifier collision must not silently replace a CV."""

    storage = build_storage(
        tmp_path
    )

    storage_key = storage.save(
        user_id="USER-001",
        document_id="DOC-001",
        document_format=CareerDocumentFormat.PDF,
        data=b"original",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        storage.save(
            user_id="USER-001",
            document_id="DOC-001",
            document_format=CareerDocumentFormat.PDF,
            data=b"replacement",
        )

    assert storage.read(
        user_id="USER-001",
        storage_key=storage_key,
    ) == b"original"