"""SQLAlchemy models for CV ingestion and evidence-review persistence."""

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

from careerops_agent_engine.infrastructure.database.base import Base

JSON_DOCUMENT = JSON().with_variant(
    JSONB,
    "postgresql",
)

NULLABLE_JSON_DOCUMENT = JSON(none_as_null=True).with_variant(
    JSONB(none_as_null=True),
    "postgresql",
)


class CareerDocumentRecord(Base):
    """Persistent metadata for one securely stored career document."""

    __tablename__ = "career_documents"

    __table_args__ = (
        CheckConstraint(
            "document_format IN ('pdf', 'docx')",
            name="document_format_valid",
        ),
        CheckConstraint(
            ("status IN ('uploaded', 'extracted', 'quarantined')"),
            name="status_valid",
        ),
        CheckConstraint(
            "size_bytes > 0",
            name="size_bytes_positive",
        ),
        UniqueConstraint(
            "storage_key",
            name="storage_key_unique",
        ),
        Index(
            "ix_career_documents_user_status",
            "user_id",
            "status",
        ),
        Index(
            "ix_career_documents_user_sha256",
            "user_id",
            "sha256_hex",
        ),
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    document_format: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    media_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    sha256_hex: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
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


class CVEvidenceReviewRunRecord(Base):
    """Latest business snapshot of one CV evidence-review run."""

    __tablename__ = "cv_evidence_review_runs"

    __table_args__ = (
        CheckConstraint(
            ("status IN ('awaiting_review', 'completed', 'invalid')"),
            name="status_valid",
        ),
        CheckConstraint(
            (
                "("
                "status = 'completed' "
                "AND review_result IS NOT NULL"
                ") OR ("
                "status <> 'completed' "
                "AND review_result IS NULL"
                ")"
            ),
            name="review_result_status_consistent",
        ),
        ForeignKeyConstraint(
            [
                "user_id",
                "document_id",
            ],
            [
                "career_documents.user_id",
                "career_documents.document_id",
            ],
            name="document_owner",
        ),
        Index(
            "ix_cv_evidence_review_runs_user_document",
            "user_id",
            "document_id",
        ),
        Index(
            "ix_cv_evidence_review_runs_user_status",
            "user_id",
            "status",
        ),
    )

    review_run_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    document_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    proposals: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    overlap_findings: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    document_warnings: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    review_result: Mapped[dict[str, object] | None] = mapped_column(
        NULLABLE_JSON_DOCUMENT,
        nullable=True,
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


class CVEvidenceReviewHistoryRecord(Base):
    """Append-only record of one applied evidence-review decision."""

    __tablename__ = "cv_evidence_review_history"

    __table_args__ = (
        UniqueConstraint(
            "review_run_id",
            "sequence_number",
            name="review_run_sequence",
        ),
        Index(
            "ix_cv_evidence_review_history_run_sequence",
            "review_run_id",
            "sequence_number",
        ),
    )

    review_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    review_run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("cv_evidence_review_runs.review_run_id"),
        nullable=False,
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    decision_payload: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    result_payload: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
