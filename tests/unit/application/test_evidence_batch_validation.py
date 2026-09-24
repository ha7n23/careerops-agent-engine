"""Tests for deterministic batched evidence validation."""

from unittest.mock import Mock

import pytest

from careerops_agent_engine.application.exceptions import (
    EvidenceDiscoveryValidationError,
)
from careerops_agent_engine.application.services.evidence_validation import (
    validate_evidence_discovery_batch,
)
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement


def build_requirement(requirement_id: str) -> JobRequirement:
    """Create one test requirement."""

    return JobRequirement(
        requirement_id=requirement_id,
        name=f"Requirement {requirement_id}",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected="Approved engineering evidence.",
        importance_score=5,
        source_text="Engineering experience is required.",
    )


def build_no_match(requirement_id: str) -> EvidenceMatch:
    """Create one valid evidence gap."""

    return EvidenceMatch(
        requirement_id=requirement_id,
        match_strength=MatchStrength.NONE,
        direct_evidence_ids=[],
        related_evidence_ids=[],
        explanation="No matching approved evidence exists.",
        gap=True,
    )


def test_batch_validation_restores_requirement_order() -> None:
    """Validated output should follow the original request order."""

    repository = Mock()

    matches = validate_evidence_discovery_batch(
        requirements=[
            build_requirement("REQ-001"),
            build_requirement("REQ-002"),
        ],
        matches=[
            build_no_match("REQ-002"),
            build_no_match("REQ-001"),
        ],
        called_tools=["search_approved_evidence"],
        observed_evidence_ids=[],
        repository=repository,
        user_id="USER-TEST",
    )

    assert [match.requirement_id for match in matches] == ["REQ-001", "REQ-002"]

    repository.get_approved.assert_not_called()


def test_batch_validation_rejects_missing_matches() -> None:
    """Every requested requirement must receive a match."""

    with pytest.raises(
        EvidenceDiscoveryValidationError,
        match="Missing: REQ-002",
    ):
        validate_evidence_discovery_batch(
            requirements=[
                build_requirement("REQ-001"),
                build_requirement("REQ-002"),
            ],
            matches=[build_no_match("REQ-001")],
            called_tools=["search_approved_evidence"],
            observed_evidence_ids=[],
            repository=Mock(),
            user_id="USER-TEST",
        )


def test_batch_validation_rejects_unknown_matches() -> None:
    """The model cannot introduce an unrequested requirement."""

    with pytest.raises(
        EvidenceDiscoveryValidationError,
        match="outside the requested requirement set: REQ-999",
    ):
        validate_evidence_discovery_batch(
            requirements=[build_requirement("REQ-001")],
            matches=[
                build_no_match("REQ-001"),
                build_no_match("REQ-999"),
            ],
            called_tools=["search_approved_evidence"],
            observed_evidence_ids=[],
            repository=Mock(),
            user_id="USER-TEST",
        )


def test_batch_validation_rejects_duplicate_matches() -> None:
    """The model cannot return two decisions for one requirement."""

    with pytest.raises(
        EvidenceDiscoveryValidationError,
        match="duplicate matches for requirement: REQ-001",
    ):
        validate_evidence_discovery_batch(
            requirements=[build_requirement("REQ-001")],
            matches=[
                build_no_match("REQ-001"),
                build_no_match("REQ-001"),
            ],
            called_tools=["search_approved_evidence"],
            observed_evidence_ids=[],
            repository=Mock(),
            user_id="USER-TEST",
        )
