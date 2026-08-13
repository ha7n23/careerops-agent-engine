"""Private local-filesystem storage for generated CV artifacts."""

import os
import re
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import uuid4

from careerops_agent_engine.application.ports.artifact_storage import (
    ArtifactStorageWriteResult,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
)

SAFE_STORAGE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LocalArtifactStorage:
    """Store generated CV artifacts under a private application root."""

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        """Create and resolve the private artifact root."""

        self._root = Path(root).expanduser().resolve()

        self._root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        artifact_format: CVArtifactFormat,
        data: bytes,
    ) -> ArtifactStorageWriteResult:
        """Atomically store one generated artifact idempotently."""

        if not data:
            raise ValueError("Generated artifact bytes cannot be empty.")

        validate_storage_identifier(
            value=cv_version_id,
            label="CV version",
        )

        validate_storage_identifier(
            value=artifact_id,
            label="Artifact",
        )

        storage_key = build_artifact_storage_key(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact_id=artifact_id,
            artifact_format=artifact_format,
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
            existing = target.read_bytes()

            if existing == data:
                return ArtifactStorageWriteResult(
                    storage_key=storage_key,
                    created=False,
                )

            raise FileExistsError(
                "Artifact storage target already exists with different bytes."
            )

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

        return ArtifactStorageWriteResult(
            storage_key=storage_key,
            created=True,
        )

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Read bytes only from the authenticated user namespace."""

        target = self._resolve_owned_path(
            user_id=user_id,
            storage_key=storage_key,
        )

        try:
            return target.read_bytes()

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "The requested CV artifact is unavailable."
            ) from exc

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Delete bytes only from the authenticated user namespace."""

        target = self._resolve_owned_path(
            user_id=user_id,
            storage_key=storage_key,
        )

        try:
            target.unlink()

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "The requested CV artifact is unavailable."
            ) from exc

    def _resolve_owned_path(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> Path:
        """Resolve one key while preventing traversal and cross-user access."""

        expected_namespace = build_artifact_user_namespace(user_id)

        if not storage_key or "\\" in storage_key:
            raise FileNotFoundError("The requested CV artifact is unavailable.")

        key_path = PurePosixPath(storage_key)

        parts = key_path.parts

        if (
            key_path.is_absolute()
            or ".." in parts
            or len(parts) != 3
            or parts[0] != expected_namespace
        ):
            raise FileNotFoundError("The requested CV artifact is unavailable.")

        target = self._root.joinpath(*parts).resolve(strict=False)

        if not target.is_relative_to(self._root):
            raise FileNotFoundError("The requested CV artifact is unavailable.")

        return target


def validate_storage_identifier(
    *,
    value: str,
    label: str,
) -> None:
    """Reject identifiers capable of influencing filesystem paths."""

    if SAFE_STORAGE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} identifier contains unsafe characters.")


def build_artifact_user_namespace(
    user_id: str,
) -> str:
    """Build one opaque user namespace."""

    if not user_id:
        raise ValueError("Artifact storage requires a user identifier.")

    digest = sha256(user_id.encode("utf-8")).hexdigest()[:32]

    return f"usr-{digest}"


def build_artifact_storage_key(
    *,
    user_id: str,
    cv_version_id: str,
    artifact_id: str,
    artifact_format: CVArtifactFormat,
) -> str:
    """Build an opaque relative artifact key."""

    namespace = build_artifact_user_namespace(user_id)

    return f"{namespace}/{cv_version_id}/{artifact_id}.{artifact_format.value}"
