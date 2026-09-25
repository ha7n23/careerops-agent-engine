"""SQLAlchemy model for persistent career evidence."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from careerops_agent_engine.infrastructure.database.base import Base

JSON_DOCUMENT = JSON().with_variant(
    JSONB,
    "postgresql",
)


class CareerEvidenceRecord(Base):
    """Persistent, user-scoped career evidence record."""

    __tablename__ = "career_evidence"

    __table_args__ = (
        CheckConstraint(
            (
                "category IN ("
                "'project', "
                "'employment', "
                "'education', "
                "'certification', "
                "'achievement', "
                "'skill'"
                ")"
            ),
            name="category_valid",
        ),
        CheckConstraint(
            (
                "verification_status IN ("
                "'pending', "
                "'approved', "
                "'rejected', "
                "'superseded'"
                ")"
            ),
            name="verification_status_valid",
        ),
        CheckConstraint(
            "lifecycle_status IN ('active', 'archived')",
            name="lifecycle_status_valid",
        ),
        CheckConstraint(
            "(lifecycle_status = 'active' AND archived_at IS NULL) OR "
            "(lifecycle_status = 'archived' AND archived_at IS NOT NULL)",
            name="lifecycle_archive_timestamp_consistent",
        ),
        Index(
            "ix_career_evidence_user_status_lifecycle",
            "user_id",
            "verification_status",
            "lifecycle_status",
        ),
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    evidence_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    category: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(250),
        nullable=False,
    )
    verification_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    lifecycle_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'active'"),
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    technologies: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    capabilities: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    approved_claims: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    source_references: Mapped[list[dict[str, object]]] = mapped_column(
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


class CareerEvidenceHistoryRecord(Base):
    """Append-only audit entry for an Evidence Registry mutation."""

    __tablename__ = "career_evidence_history"

    __table_args__ = (
        CheckConstraint(
            "action IN ('edit', 'archive', 'restore')",
            name="action_valid",
        ),
        ForeignKeyConstraint(
            ["user_id", "evidence_id"],
            ["career_evidence.user_id", "career_evidence.evidence_id"],
            ondelete="RESTRICT",
        ),
        Index(
            "ix_career_evidence_history_user_evidence_created",
            "user_id",
            "evidence_id",
            "created_at",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    evidence_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    before_payload: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    after_payload: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
