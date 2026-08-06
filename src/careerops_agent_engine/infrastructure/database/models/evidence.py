"""SQLAlchemy model for persistent career evidence."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Index,
    String,
    func,
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
        Index(
            "ix_career_evidence_user_status",
            "user_id",
            "verification_status",
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
