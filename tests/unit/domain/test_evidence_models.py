"""Tests for career-evidence domain models."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import MatchStrength
from careerops_agent_engine.domain.models.evidence import EvidenceMatch


def test_strong_match_requires_direct_evidence() -> None:
    """Strong evidence must reference at least one direct record."""

    with pytest.raises(
        ValidationError,
        match="A strong match requires direct evidence",
    ):
        EvidenceMatch(
            requirement_id="REQ-001",
            match_strength=MatchStrength.STRONG,
            direct_evidence_ids=[],
            related_evidence_ids=[],
            explanation="The requirement is fully supported.",
            gap=False,
        )


def test_related_match_preserves_skill_gap() -> None:
    """Related experience must not be represented as direct experience."""

    match = EvidenceMatch(
        requirement_id="REQ-002",
        match_strength=MatchStrength.RELATED,
        direct_evidence_ids=[],
        related_evidence_ids=["EVD-DOCKER-001"],
        explanation=(
            "Docker is related to containerisation but does not prove "
            "Kubernetes implementation."
        ),
        gap=True,
    )

    assert match.direct_evidence_ids == []
    assert match.related_evidence_ids == ["EVD-DOCKER-001"]
    assert match.gap is True


def test_no_match_cannot_reference_evidence() -> None:
    """A no-match outcome must not hide evidence references inside it."""

    with pytest.raises(
        ValidationError,
        match="A no-match result cannot reference evidence",
    ):
        EvidenceMatch(
            requirement_id="REQ-003",
            match_strength=MatchStrength.NONE,
            direct_evidence_ids=[],
            related_evidence_ids=["EVD-001"],
            explanation="No direct implementation evidence exists.",
            gap=True,
        )


def test_direct_and_related_evidence_must_be_disjoint() -> None:
    """One evidence record cannot have conflicting match semantics."""

    with pytest.raises(
        ValidationError,
        match="cannot be both direct and related",
    ):
        EvidenceMatch(
            requirement_id="REQ-004",
            match_strength=MatchStrength.STRONG,
            direct_evidence_ids=["EVD-001"],
            related_evidence_ids=["EVD-001"],
            explanation="Conflicting evidence classification.",
            gap=False,
        )
