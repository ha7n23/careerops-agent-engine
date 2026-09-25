"""add evidence lifecycle and audit

Revision ID: a41b8f2c7d90
Revises: c766dc2b639a
Create Date: 2026-09-25 12:15:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a41b8f2c7d90"
down_revision: str | Sequence[str] | None = "c766dc2b639a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add recoverable lifecycle state and append-only mutation history."""

    op.add_column(
        "career_evidence",
        sa.Column(
            "lifecycle_status",
            sa.String(length=16),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
    )
    op.add_column(
        "career_evidence",
        sa.Column(
            "archived_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        op.f("ck_career_evidence_lifecycle_status_valid"),
        "career_evidence",
        "lifecycle_status IN ('active', 'archived')",
    )
    op.create_check_constraint(
        op.f("ck_career_evidence_lifecycle_archive_timestamp_consistent"),
        "career_evidence",
        "(lifecycle_status = 'active' AND archived_at IS NULL) OR "
        "(lifecycle_status = 'archived' AND archived_at IS NOT NULL)",
    )
    op.drop_index(
        "ix_career_evidence_user_status",
        table_name="career_evidence",
    )
    op.create_index(
        "ix_career_evidence_user_status_lifecycle",
        "career_evidence",
        ["user_id", "verification_status", "lifecycle_status"],
        unique=False,
    )

    op.create_table(
        "career_evidence_history",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("evidence_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column(
            "before_payload",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "after_payload",
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
            "action IN ('edit', 'archive', 'restore')",
            name=op.f("ck_career_evidence_history_action_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "evidence_id"],
            ["career_evidence.user_id", "career_evidence.evidence_id"],
            name=op.f("fk_career_evidence_history_user_id_career_evidence"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "event_id",
            name=op.f("pk_career_evidence_history"),
        ),
    )
    op.create_index(
        "ix_career_evidence_history_user_evidence_created",
        "career_evidence_history",
        ["user_id", "evidence_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove evidence lifecycle state and mutation history."""

    op.drop_index(
        "ix_career_evidence_history_user_evidence_created",
        table_name="career_evidence_history",
    )
    op.drop_table("career_evidence_history")

    op.drop_index(
        "ix_career_evidence_user_status_lifecycle",
        table_name="career_evidence",
    )
    op.create_index(
        "ix_career_evidence_user_status",
        "career_evidence",
        ["user_id", "verification_status"],
        unique=False,
    )
    op.drop_constraint(
        op.f("ck_career_evidence_lifecycle_archive_timestamp_consistent"),
        "career_evidence",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_career_evidence_lifecycle_status_valid"),
        "career_evidence",
        type_="check",
    )
    op.drop_column("career_evidence", "archived_at")
    op.drop_column("career_evidence", "lifecycle_status")
