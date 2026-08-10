"""SQLAlchemy persistence models for versioned CV artifacts."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from careerops_agent_engine.infrastructure.database.base import (
    Base,
)

JSON_DOCUMENT = JSON().with_variant(
    JSONB,
    "postgresql",
)


class CVVersionRecord(Base):
    """Persistent content and provenance for one CV version."""

    __tablename__ = "cv_versions"

    __table_args__ = (
        CheckConstraint(
            "version_number >= 1",
            name="version_number_positive",
        ),
        CheckConstraint(
            "status IN ('assembled', 'rendered', 'verified')",
            name="status_valid",
        ),
        UniqueConstraint(
            "user_id",
            "cv_id",
            "version_number",
            name="user_cv_version_number",
        ),
        ForeignKeyConstraint(
            [
                "user_id",
                "source_document_id",
            ],
            [
                "career_documents.user_id",
                "career_documents.document_id",
            ],
            name="source_document_owner",
        ),
        Index(
            "ix_cv_versions_user_cv",
            "user_id",
            "cv_id",
        ),
        Index(
            "ix_cv_versions_user_job",
            "user_id",
            "job_id",
        ),
        Index(
            "ix_cv_versions_user_status",
            "user_id",
            "status",
        ),
    )

    cv_version_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    cv_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    parent_version_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("cv_versions.cv_version_id"),
        nullable=True,
    )

    source_document_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    job_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    thread_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("job_analysis_runs.thread_id"),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    structured_cv: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    applied_changes: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    provenance: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class CVArtifactRecord(Base):
    """Persistent metadata for one rendered CV artifact."""

    __tablename__ = "cv_artifacts"

    __table_args__ = (
        CheckConstraint(
            "artifact_format IN ('docx', 'pdf')",
            name="artifact_format_valid",
        ),
        CheckConstraint(
            ("verification_status IN ('pending', 'verified', 'failed')"),
            name="verification_status_valid",
        ),
        CheckConstraint(
            "size_bytes > 0",
            name="size_bytes_positive",
        ),
        UniqueConstraint(
            "cv_version_id",
            "artifact_format",
            name="version_artifact_format",
        ),
        UniqueConstraint(
            "storage_key",
            name="cv_artifact_storage_key_unique",
        ),
        Index(
            "ix_cv_artifacts_version_format",
            "cv_version_id",
            "artifact_format",
        ),
    )

    artifact_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    cv_version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("cv_versions.cv_version_id"),
        nullable=False,
    )

    artifact_format: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    sha256_hex: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    verification_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    verification_notes: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
