"""Application port for generated CV artifact byte storage."""

from dataclasses import dataclass
from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
)


@dataclass(frozen=True)
class ArtifactStorageWriteResult:
    """Result of one idempotent artifact-storage write."""

    storage_key: str
    created: bool


class ArtifactStorage(Protocol):
    """Store generated CV artifacts behind a private boundary."""

    def save(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        artifact_format: CVArtifactFormat,
        data: bytes,
    ) -> ArtifactStorageWriteResult:
        """Persist bytes idempotently and return their opaque key."""

        ...

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Read generated artifact bytes inside the user boundary."""

        ...

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Delete artifact bytes during compensating cleanup."""

        ...
