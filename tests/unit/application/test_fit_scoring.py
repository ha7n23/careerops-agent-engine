"""Tests for deterministic job-fit scoring."""

import pytest

from careerops_agent_engine.application.services.fit_scoring import (
    calculate_evidence_weighted_fit,
    calculate_weighted_fit,
)
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
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


def build_match(
    *,
    requirement_id: str,
    strength: MatchStrength,
) -> EvidenceMatch:
    """Create a valid evidence match for a selected strength."""

    if strength is MatchStrength.STRONG:
        return EvidenceMatch(
            requirement_id=requirement_id,
            match_strength=strength,
            direct_evidence_ids=[f"EVD-{requirement_id}"],
            related_evidence_ids=[],
            explanation="Direct approved evidence exists.",
            gap=False,
        )

    if strength is MatchStrength.PARTIAL:
        return EvidenceMatch(
            requirement_id=requirement_id,
            match_strength=strength,
            direct_evidence_ids=[f"EVD-{requirement_id}"],
            related_evidence_ids=[],
            explanation=(
                "Direct evidence exists but does not fully satisfy the requirement."
            ),
            gap=True,
        )

    if strength is MatchStrength.RELATED:
        return EvidenceMatch(
            requirement_id=requirement_id,
            match_strength=strength,
            direct_evidence_ids=[],
            related_evidence_ids=[f"EVD-{requirement_id}"],
            explanation="Only adjacent experience exists.",
            gap=True,
        )

    return EvidenceMatch(
        requirement_id=requirement_id,
        match_strength=MatchStrength.NONE,
        direct_evidence_ids=[],
        related_evidence_ids=[],
        explanation="No supporting evidence exists.",
        gap=True,
    )


def test_all_requirement_ids_matched_returns_full_score() -> None:
    """The original direct-ID function should still return 100 percent."""

    score = calculate_weighted_fit(
        requirements=build_requirements(),
        matched_requirement_ids={
            "REQ-PYTHON",
            "REQ-LANGGRAPH",
        },
    )

    assert score == 100.0


def test_original_partial_id_match_uses_requirement_weights() -> None:
    """A Python-only direct-ID match should use five of nine weight."""

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


def test_evidence_scoring_uses_strength_factors() -> None:
    """Strong receives full credit and partial receives half credit."""

    score = calculate_evidence_weighted_fit(
        requirements=build_requirements(),
        evidence_matches=[
            build_match(
                requirement_id="REQ-PYTHON",
                strength=MatchStrength.STRONG,
            ),
            build_match(
                requirement_id="REQ-LANGGRAPH",
                strength=MatchStrength.PARTIAL,
            ),
        ],
    )

    # Python earns 5 and LangGraph earns 2 from its weight of 4.
    assert score == 77.78


def test_related_evidence_receives_no_direct_fit_credit() -> None:
    """Adjacent skills must not silently become direct experience."""

    score = calculate_evidence_weighted_fit(
        requirements=build_requirements(),
        evidence_matches=[
            build_match(
                requirement_id="REQ-PYTHON",
                strength=MatchStrength.STRONG,
            ),
            build_match(
                requirement_id="REQ-LANGGRAPH",
                strength=MatchStrength.RELATED,
            ),
        ],
    )

    assert score == 55.56


def test_missing_evidence_match_is_rejected() -> None:
    """Every extracted requirement must receive one match outcome."""

    with pytest.raises(
        ValueError,
        match="missing requirements",
    ):
        calculate_evidence_weighted_fit(
            requirements=build_requirements(),
            evidence_matches=[
                build_match(
                    requirement_id="REQ-PYTHON",
                    strength=MatchStrength.STRONG,
                )
            ],
        )


def test_duplicate_evidence_matches_are_rejected() -> None:
    """One requirement cannot receive conflicting duplicate matches."""

    duplicate_match = build_match(
        requirement_id="REQ-PYTHON",
        strength=MatchStrength.STRONG,
    )

    with pytest.raises(
        ValueError,
        match="unique requirement identifiers",
    ):
        calculate_evidence_weighted_fit(
            requirements=build_requirements(),
            evidence_matches=[
                duplicate_match,
                duplicate_match,
                build_match(
                    requirement_id="REQ-LANGGRAPH",
                    strength=MatchStrength.NONE,
                ),
            ],
        )
