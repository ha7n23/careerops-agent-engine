"""Tests for job-analysis domain models."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)


def build_requirement(
    requirement_id: str = "REQ-001",
) -> JobRequirement:
    """Create a valid job requirement for tests."""

    return JobRequirement(
        requirement_id=requirement_id,
        name="LangGraph",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected="Implementation of stateful agent workflows.",
        importance_score=5,
        source_text="Experience building stateful workflows using LangGraph.",
    )


def test_job_requirement_accepts_valid_data() -> None:
    """A complete requirement should pass validation."""

    requirement = build_requirement()

    assert requirement.name == "LangGraph"
    assert requirement.category is RequirementCategory.ESSENTIAL
    assert requirement.importance_score == 5


@pytest.mark.parametrize("score", [0, 6])
def test_job_requirement_rejects_invalid_importance_score(
    score: int,
) -> None:
    """Importance must remain inside the supported one-to-five range."""

    with pytest.raises(ValidationError):
        JobRequirement(
            requirement_id="REQ-001",
            name="LangGraph",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected="Stateful workflow implementation.",
            importance_score=score,
            source_text="LangGraph experience is required.",
        )


def test_job_requirement_extraction_rejects_duplicate_ids() -> None:
    """Two extracted requirements cannot share an identifier."""

    with pytest.raises(
        ValidationError,
        match="Requirement identifiers must be unique",
    ):
        JobRequirementExtraction(
            role_title="Junior AI Engineer",
            requirements=[
                build_requirement("REQ-001"),
                build_requirement("REQ-001"),
            ],
        )


def test_domain_models_reject_unknown_fields() -> None:
    """Unexpected model output must not silently enter domain data."""

    with pytest.raises(ValidationError):
        JobRequirement.model_validate(
            {
                "requirement_id": "REQ-001",
                "name": "LangGraph",
                "category": "essential",
                "evidence_expected": "Stateful workflow implementation.",
                "importance_score": 5,
                "source_text": "LangGraph is required.",
                "invented_field": "unexpected",
            }
        )
