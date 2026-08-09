"""Tests for CareerOps SQLAlchemy table definitions."""

from sqlalchemy import create_engine, inspect

from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.models.cv_evidence import (
    CareerDocumentRecord,
    CVEvidenceReviewHistoryRecord,
    CVEvidenceReviewRunRecord,
)
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceRecord,
)
from careerops_agent_engine.infrastructure.database.models.job_analysis import (
    CVReviewHistoryRecord,
    JobAnalysisRunRecord,
)


def test_career_evidence_table_has_expected_primary_key() -> None:
    """Evidence should be uniquely scoped by user and evidence ID."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        assert CareerEvidenceRecord.__tablename__ in inspector.get_table_names()

        primary_key = inspector.get_pk_constraint(CareerEvidenceRecord.__tablename__)

        assert primary_key["constrained_columns"] == [
            "user_id",
            "evidence_id",
        ]
    finally:
        engine.dispose()


def test_career_evidence_table_has_integrity_constraints() -> None:
    """The database should constrain domain enumeration values."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        constraints = inspector.get_check_constraints(
            CareerEvidenceRecord.__tablename__
        )
        constraint_names = {constraint["name"] for constraint in constraints}

        assert "ck_career_evidence_category_valid" in constraint_names
        assert "ck_career_evidence_verification_status_valid" in constraint_names
    finally:
        engine.dispose()


def test_job_analysis_audit_tables_are_registered() -> None:
    """Business run and review-history tables should exist."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        table_names = set(inspector.get_table_names())

        assert JobAnalysisRunRecord.__tablename__ in table_names
        assert CVReviewHistoryRecord.__tablename__ in table_names
    finally:
        engine.dispose()


def test_job_analysis_run_has_integrity_constraints() -> None:
    """Run snapshots should constrain status and fit score."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        constraints = inspector.get_check_constraints(
            JobAnalysisRunRecord.__tablename__
        )

        names = {constraint["name"] for constraint in constraints}

        assert "ck_job_analysis_runs_status_valid" in names
        assert "ck_job_analysis_runs_review_status_valid" in names
        assert "ck_job_analysis_runs_fit_score_range" in names
    finally:
        engine.dispose()


def test_review_history_has_unique_thread_sequence() -> None:
    """One review sequence may appear only once per thread."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        constraints = inspector.get_unique_constraints(
            CVReviewHistoryRecord.__tablename__
        )

        constrained_columns = {
            tuple(constraint["column_names"]) for constraint in constraints
        }

        assert (
            "thread_id",
            "sequence_number",
        ) in constrained_columns
    finally:
        engine.dispose()


def test_cv_evidence_persistence_tables_are_registered() -> None:
    """CV document and evidence-review business tables should exist."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)

        inspector = inspect(engine)

        table_names = set(inspector.get_table_names())

        assert CareerDocumentRecord.__tablename__ in table_names

        assert CVEvidenceReviewRunRecord.__tablename__ in table_names

        assert CVEvidenceReviewHistoryRecord.__tablename__ in table_names

    finally:
        engine.dispose()


def test_career_document_table_has_integrity_constraints() -> None:
    """Stored CV metadata should retain basic database invariants."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)

        inspector = inspect(engine)

        primary_key = inspector.get_pk_constraint(CareerDocumentRecord.__tablename__)

        assert primary_key["constrained_columns"] == [
            "user_id",
            "document_id",
        ]

        constraints = inspector.get_check_constraints(
            CareerDocumentRecord.__tablename__
        )

        names = {constraint["name"] for constraint in constraints}

        assert "ck_career_documents_document_format_valid" in names

        assert "ck_career_documents_status_valid" in names

        assert "ck_career_documents_size_bytes_positive" in names

    finally:
        engine.dispose()


def test_cv_evidence_review_run_links_to_owned_document() -> None:
    """Review runs should reference the document within its user scope."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)

        inspector = inspect(engine)

        foreign_keys = inspector.get_foreign_keys(
            CVEvidenceReviewRunRecord.__tablename__
        )

        document_foreign_keys = [
            foreign_key
            for foreign_key in foreign_keys
            if (foreign_key["referred_table"] == CareerDocumentRecord.__tablename__)
        ]

        assert len(document_foreign_keys) == 1

        foreign_key = document_foreign_keys[0]

        assert foreign_key["constrained_columns"] == [
            "user_id",
            "document_id",
        ]

        assert foreign_key["referred_columns"] == [
            "user_id",
            "document_id",
        ]

    finally:
        engine.dispose()


def test_cv_evidence_review_history_has_unique_run_sequence() -> None:
    """One sequence number may occur only once per review run."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)

        inspector = inspect(engine)

        constraints = inspector.get_unique_constraints(
            CVEvidenceReviewHistoryRecord.__tablename__
        )

        constrained_columns = {
            tuple(constraint["column_names"]) for constraint in constraints
        }

        assert (
            "review_run_id",
            "sequence_number",
        ) in constrained_columns

    finally:
        engine.dispose()
