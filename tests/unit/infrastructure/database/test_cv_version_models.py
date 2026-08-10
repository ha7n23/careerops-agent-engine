"""Tests for CV-version SQLAlchemy metadata."""

from typing import cast

from sqlalchemy import (
    CheckConstraint,
    Table,
    UniqueConstraint,
)

from careerops_agent_engine.infrastructure.database.models.cv_version import (
    CVArtifactRecord,
    CVVersionRecord,
)


def constraint_names(
    model: type[CVVersionRecord] | type[CVArtifactRecord],
    constraint_type: type[CheckConstraint] | type[UniqueConstraint],
) -> set[str]:
    """Return named constraints of one SQLAlchemy type."""

    table = cast(
        Table,
        model.__table__,
    )

    return {
        name
        for constraint in table.constraints
        if isinstance(
            constraint,
            constraint_type,
        )
        and isinstance(
            name := constraint.name,
            str,
        )
    }


def test_cv_version_table_has_expected_identity_constraints() -> None:
    """CV versions should preserve family/version identity."""

    table = cast(
        Table,
        CVVersionRecord.__table__,
    )

    assert table.name == "cv_versions"

    assert {column.name for column in table.primary_key.columns} == {"cv_version_id"}

    names = constraint_names(
        CVVersionRecord,
        UniqueConstraint,
    )

    assert "user_cv_version_number" in names


def test_cv_version_table_links_source_document_and_job_run() -> None:
    """Version rows should retain their trusted workflow lineage."""

    foreign_keys = {
        foreign_key.target_fullname
        for foreign_key in CVVersionRecord.__table__.foreign_keys
    }

    assert "career_documents.user_id" in foreign_keys

    assert "career_documents.document_id" in foreign_keys

    assert "job_analysis_runs.thread_id" in foreign_keys


def test_cv_version_table_has_lifecycle_checks() -> None:
    """Database constraints should enforce basic version lifecycle."""

    names = constraint_names(
        CVVersionRecord,
        CheckConstraint,
    )

    assert "ck_cv_versions_version_number_positive" in names

    assert "ck_cv_versions_status_valid" in names


def test_cv_artifact_table_enforces_one_file_per_format() -> None:
    """One CV version cannot own two DOCX or two PDF artifacts."""

    names = constraint_names(
        CVArtifactRecord,
        UniqueConstraint,
    )

    assert "version_artifact_format" in names

    assert "cv_artifact_storage_key_unique" in names


def test_cv_artifact_table_has_file_validation_constraints() -> None:
    """Artifact metadata should enforce supported lifecycle values."""

    names = constraint_names(
        CVArtifactRecord,
        CheckConstraint,
    )

    assert "ck_cv_artifacts_artifact_format_valid" in names

    assert "ck_cv_artifacts_verification_status_valid" in names

    assert "ck_cv_artifacts_size_bytes_positive" in names
