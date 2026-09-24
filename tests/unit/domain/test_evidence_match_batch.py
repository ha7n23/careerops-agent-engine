"""Tests for batched evidence-match domain output."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import MatchStrength
from careerops_agent_engine.domain.models.evidence import (
    EvidenceMatch,
    EvidenceMatchBatch,
)


def build_no_match(requirement_id: str) -> EvidenceMatch:
    """Create one valid deterministic gap."""

    return EvidenceMatch(
        requirement_id=requirement_id,
        match_strength=MatchStrength.NONE,
        direct_evidence_ids=[],
        related_evidence_ids=[],
        explanation="No matching approved evidence exists.",
        gap=True,
    )


def test_batch_accepts_unique_requirement_matches() -> None:
    """A valid batch may contain one match per requirement."""

    batch = EvidenceMatchBatch(
        matches=[
            build_no_match("REQ-001"),
            build_no_match("REQ-002"),
        ]
    )

    assert [match.requirement_id for match in batch.matches] == ["REQ-001", "REQ-002"]


def test_batch_rejects_duplicate_requirement_matches() -> None:
    """One requirement must not receive multiple model decisions."""

    with pytest.raises(
        ValidationError,
        match="unique requirement identifiers",
    ):
        EvidenceMatchBatch(
            matches=[
                build_no_match("REQ-001"),
                build_no_match("REQ-001"),
            ]
        )


def test_batch_rejects_empty_model_output() -> None:
    """A batch response must contain at least one match."""

    with pytest.raises(ValidationError):
        EvidenceMatchBatch(matches=[])
