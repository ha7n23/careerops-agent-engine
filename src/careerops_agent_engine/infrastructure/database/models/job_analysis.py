"""SQLAlchemy models for job-analysis business audit history."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
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


class JobAnalysisRunRecord(Base):
    """Latest persistent business snapshot of one workflow run."""

    __tablename__ = "job_analysis_runs"

    __table_args__ = (
        CheckConstraint(
            ("status IN ('awaiting_review', 'completed', 'invalid')"),
            name="status_valid",
        ),
        CheckConstraint(
            (
                "review_status IS NULL OR "
                "review_status IN ("
                "'not_requested', "
                "'pending', "
                "'approved', "
                "'edited', "
                "'rejected', "
                "'regeneration_requested'"
                ")"
            ),
            name="review_status_valid",
        ),
        CheckConstraint(
            ("fit_score IS NULL OR (fit_score >= 0 AND fit_score <= 100)"),
            name="fit_score_range",
        ),
        Index(
            "ix_job_analysis_runs_user_job",
            "user_id",
            "job_id",
        ),
        Index(
            "ix_job_analysis_runs_user_status",
            "user_id",
            "status",
        ),
    )

    thread_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    job_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    role_title: Mapped[str | None] = mapped_column(
        String(250),
        nullable=True,
    )
    fit_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    review_status: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    cv_proposals: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    claim_verification_reports: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    reviewable_proposal_ids: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )
    blocked_proposal_ids: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    final_cv_proposals: Mapped[list[dict[str, object]]] = mapped_column(
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


class CVReviewHistoryRecord(Base):
    """Append-only record of one applied human review decision."""

    __tablename__ = "cv_review_history"

    __table_args__ = (
        CheckConstraint(
            ("action IN ('approve', 'edit', 'reject', 'regenerate')"),
            name="action_valid",
        ),
        CheckConstraint(
            ("result_status IN ('awaiting_review', 'completed')"),
            name="result_status_valid",
        ),
        CheckConstraint(
            (
                "result_review_status IS NULL OR "
                "result_review_status IN ("
                "'not_requested', "
                "'pending', "
                "'approved', "
                "'edited', "
                "'rejected', "
                "'regeneration_requested'"
                ")"
            ),
            name="result_review_status_valid",
        ),
        UniqueConstraint(
            "thread_id",
            "sequence_number",
            name="thread_sequence",
        ),
        Index(
            "ix_cv_review_history_thread_sequence",
            "thread_id",
            "sequence_number",
        ),
    )

    review_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    thread_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("job_analysis_runs.thread_id"),
        nullable=False,
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    decision_payload: Mapped[dict[str, object]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    result_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    result_review_status: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    resulting_cv_proposals: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    resulting_verification_reports: Mapped[list[dict[str, object]]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
