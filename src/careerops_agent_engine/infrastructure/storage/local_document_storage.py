"""Private local-filesystem storage for career documents."""

import os
import re
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import uuid4

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)

SAFE_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

STORAGE_SUFFIXES = {
    CareerDocumentFormat.PDF: ".pdf",
    CareerDocumentFormat.DOCX: ".docx",
    CareerDocumentFormat.TEXT: ".txt",
}


class LocalDocumentStorage:
    """Store career-document bytes under a private application root."""

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        """Create and resolve the private storage root."""

        self._root = Path(root).expanduser().resolve()

        self._root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Atomically persist bytes and return an opaque storage key."""

        validate_document_id(document_id)

        storage_key = build_storage_key(
            user_id=user_id,
            document_id=document_id,
            document_format=document_format,
        )

        target = self._resolve_owned_path(
            user_id=user_id,
            storage_key=storage_key,
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if target.exists():
            raise FileExistsError("Document storage target already exists.")

        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(
                temporary,
                target,
            )
        finally:
            if temporary.exists():
                temporary.unlink()

        return storage_key

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Read bytes only from the authenticated user's namespace."""

        target = self._resolve_owned_path(
            user_id=user_id,
            storage_key=storage_key,
        )

        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError("The requested document is unavailable.") from exc

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Delete bytes only from the authenticated user's namespace."""

        target = self._resolve_owned_path(
            user_id=user_id,
            storage_key=storage_key,
        )

        try:
            target.unlink()
        except FileNotFoundError as exc:
            raise FileNotFoundError("The requested document is unavailable.") from exc

    def _resolve_owned_path(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> Path:
        """Resolve one opaque key while preventing path traversal."""

        expected_namespace = build_user_namespace(user_id)

        if not storage_key or "\\" in storage_key:
            raise FileNotFoundError("The requested document is unavailable.")

        key_path = PurePosixPath(storage_key)

        parts = key_path.parts

        if (
            key_path.is_absolute()
            or ".." in parts
            or len(parts) != 3
            or parts[0] != "documents"
            or parts[1] != expected_namespace
        ):
            raise FileNotFoundError("The requested document is unavailable.")

        target = self._root.joinpath(*parts).resolve(strict=False)

        if not target.is_relative_to(self._root):
            raise FileNotFoundError("The requested document is unavailable.")

        return target


def validate_document_id(
    document_id: str,
) -> None:
    """Reject identifiers that could influence filesystem paths."""

    if SAFE_DOCUMENT_ID.fullmatch(document_id) is None:
        raise ValueError("Document identifier contains unsafe characters.")


def build_user_namespace(
    user_id: str,
) -> str:
    """Create a stable opaque directory for one user."""

    if not user_id:
        raise ValueError("Document storage requires a user identifier.")

    digest = sha256(user_id.encode("utf-8")).hexdigest()[:32]

    return f"usr-{digest}"


def build_storage_key(
    *,
    user_id: str,
    document_id: str,
    document_format: CareerDocumentFormat,
) -> str:
    """Build an opaque relative storage key."""

    namespace = build_user_namespace(user_id)

    suffix = STORAGE_SUFFIXES[document_format]

    return f"documents/{namespace}/{document_id}{suffix}"
