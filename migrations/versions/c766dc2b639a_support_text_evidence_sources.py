"""Support trusted text evidence sources.

Revision ID: c766dc2b639a
Revises: 8eb5336b0db8
Create Date: 2026-09-25

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c766dc2b639a"
down_revision: str | Sequence[str] | None = "8eb5336b0db8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT_NAME = "ck_career_documents_document_format_valid"


def upgrade() -> None:
    """Allow UTF-8 text in the shared career-document pipeline."""

    op.drop_constraint(
        op.f(CONSTRAINT_NAME),
        "career_documents",
        type_="check",
    )

    op.create_check_constraint(
        op.f(CONSTRAINT_NAME),
        "career_documents",
        "document_format IN ('pdf', 'docx', 'text')",
    )


def downgrade() -> None:
    """Restore the PDF/DOCX-only document-format constraint."""

    op.drop_constraint(
        op.f(CONSTRAINT_NAME),
        "career_documents",
        type_="check",
    )

    op.create_check_constraint(
        op.f(CONSTRAINT_NAME),
        "career_documents",
        "document_format IN ('pdf', 'docx')",
    )
