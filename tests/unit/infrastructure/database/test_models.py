"""Tests for CareerOps SQLAlchemy table definitions."""

from sqlalchemy import create_engine, inspect

from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceRecord,
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
