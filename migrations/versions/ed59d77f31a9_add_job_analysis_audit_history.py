"""add job analysis audit history

Revision ID: ed59d77f31a9
Revises: ce2e57cff0a3
Create Date: 2026-08-08 10:59:02.804949

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "ed59d77f31a9"
down_revision: str | Sequence[str] | None = "ce2e57cff0a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create CareerOps job-analysis business audit tables."""

    op.create_table(
        "job_analysis_runs",
        sa.Column(
            "thread_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "role_title",
            sa.String(length=250),
            nullable=True,
        ),
        sa.Column(
            "fit_score",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "review_status",
            sa.String(length=32),
            nullable=True,
        ),
        sa.Column(
            "cv_proposals",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "claim_verification_reports",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "reviewable_proposal_ids",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "blocked_proposal_ids",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "final_cv_proposals",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_job_analysis_runs_review_status_valid"),
        ),
        sa.CheckConstraint(
            ("status IN ('awaiting_review', 'completed', 'invalid')"),
            name=op.f("ck_job_analysis_runs_status_valid"),
        ),
        sa.CheckConstraint(
            ("fit_score IS NULL OR (fit_score >= 0 AND fit_score <= 100)"),
            name=op.f("ck_job_analysis_runs_fit_score_range"),
        ),
        sa.PrimaryKeyConstraint(
            "thread_id",
            name=op.f("pk_job_analysis_runs"),
        ),
    )

    op.create_index(
        "ix_job_analysis_runs_user_job",
        "job_analysis_runs",
        [
            "user_id",
            "job_id",
        ],
        unique=False,
    )

    op.create_index(
        "ix_job_analysis_runs_user_status",
        "job_analysis_runs",
        [
            "user_id",
            "status",
        ],
        unique=False,
    )

    op.create_table(
        "cv_review_history",
        sa.Column(
            "review_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "thread_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "sequence_number",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "action",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "decision_payload",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "result_status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "result_review_status",
            sa.String(length=32),
            nullable=True,
        ),
        sa.Column(
            "resulting_cv_proposals",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "resulting_verification_reports",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            ("action IN ('approve', 'edit', 'reject', 'regenerate')"),
            name=op.f("ck_cv_review_history_action_valid"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_cv_review_history_result_review_status_valid"),
        ),
        sa.CheckConstraint(
            ("result_status IN ('awaiting_review', 'completed')"),
            name=op.f("ck_cv_review_history_result_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["job_analysis_runs.thread_id"],
            name=op.f("fk_cv_review_history_thread_id_job_analysis_runs"),
        ),
        sa.PrimaryKeyConstraint(
            "review_id",
            name=op.f("pk_cv_review_history"),
        ),
        sa.UniqueConstraint(
            "thread_id",
            "sequence_number",
            name="thread_sequence",
        ),
    )

    op.create_index(
        "ix_cv_review_history_thread_sequence",
        "cv_review_history",
        [
            "thread_id",
            "sequence_number",
        ],
        unique=False,
    )


def downgrade() -> None:
    """Remove CareerOps job-analysis business audit tables."""

    # Child table first because it references job_analysis_runs.
    op.drop_index(
        "ix_cv_review_history_thread_sequence",
        table_name="cv_review_history",
    )
    op.drop_table("cv_review_history")

    op.drop_index(
        "ix_job_analysis_runs_user_status",
        table_name="job_analysis_runs",
    )
    op.drop_index(
        "ix_job_analysis_runs_user_job",
        table_name="job_analysis_runs",
    )
    op.drop_table("job_analysis_runs")
