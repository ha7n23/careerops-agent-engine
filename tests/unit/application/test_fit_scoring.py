"""Tests for deterministic fit scoring."""

import pytest

from careerops_agent_engine.application.services.fit_scoring import (
    calculate_weighted_fit,
)
from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.job import JobRequirement


def build_requirements() -> list[JobRequirement]:
    """Create weighted requirements for scoring tests."""

    return [
        JobRequirement(
            requirement_id="REQ-PYTHON",
            name="Python",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected="Python development experience.",
            importance_score=5,
            source_text="Python is required.",
        ),
        JobRequirement(
            requirement_id="REQ-LANGGRAPH",
            name="LangGraph",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected="Stateful agent workflow experience.",
            importance_score=4,
            source_text="LangGraph is required.",
        ),
    ]


def test_all_requirements_matched_returns_full_score() -> None:
    """Matching every weighted requirement should return 100 percent."""

    score = calculate_weighted_fit(
        requirements=build_requirements(),
        matched_requirement_ids={
            "REQ-PYTHON",
            "REQ-LANGGRAPH",
        },
    )

    assert score == 100.0


def test_partial_match_uses_requirement_weights() -> None:
    """A Python-only match should use its five-of-nine weight."""

    score = calculate_weighted_fit(
        requirements=build_requirements(),
        matched_requirement_ids={"REQ-PYTHON"},
    )

    assert score == 55.56


def test_unknown_matched_requirement_is_rejected() -> None:
    """The score must not silently accept unknown identifiers."""

    with pytest.raises(
        ValueError,
        match="Unknown matched requirement identifiers",
    ):
        calculate_weighted_fit(
            requirements=build_requirements(),
            matched_requirement_ids={"REQ-UNKNOWN"},
        )
