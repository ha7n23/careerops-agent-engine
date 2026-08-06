"""Tests for deterministic evidence-agent output validation."""

import pytest

from careerops_agent_engine.application.exceptions import (
    EvidenceDiscoveryValidationError,
)
from careerops_agent_engine.application.services.evidence_validation import (
    validate_evidence_discovery_result,
)
from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_requirement() -> JobRequirement:
    """Create a Kubernetes requirement."""

    return JobRequirement(
        requirement_id="REQ-KUBERNETES",
        name="Kubernetes",
        category=RequirementCategory.DESIRABLE,
        evidence_expected="Exposure to Kubernetes orchestration.",
        importance_score=2,
        source_text="Kubernetes exposure is desirable.",
    )


def build_repository() -> InMemoryEvidenceRepository:
    """Create approved Docker evidence for one user."""

    evidence = CareerEvidence(
        evidence_id="EVD-DOCKER",
        category=EvidenceCategory.PROJECT,
        title="Containerised AI Service",
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python", "Docker"],
        capabilities=["Application containerisation"],
        approved_claims=[
            "Containerised a FastAPI service using Docker.",
        ],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id="SRC-DOCKER",
            )
        ],
    )

    return InMemoryEvidenceRepository({"USER-001": [evidence]})


def build_related_match() -> EvidenceMatch:
    """Represent Docker as related, not direct, evidence."""

    return EvidenceMatch(
        requirement_id="REQ-KUBERNETES",
        match_strength=MatchStrength.RELATED,
        direct_evidence_ids=[],
        related_evidence_ids=["EVD-DOCKER"],
        explanation=(
            "Docker demonstrates containerisation experience but "
            "does not prove Kubernetes usage."
        ),
        gap=True,
    )


def test_related_observed_evidence_is_accepted() -> None:
    """Retrieved approved Docker evidence may support a related match."""

    validate_evidence_discovery_result(
        requirement=build_requirement(),
        match=build_related_match(),
        called_tools=["search_approved_evidence"],
        observed_evidence_ids=["EVD-DOCKER"],
        repository=build_repository(),
        user_id="USER-001",
    )


def test_agent_must_search_before_returning_match() -> None:
    """A final answer without evidence search must be rejected."""

    with pytest.raises(
        EvidenceDiscoveryValidationError,
        match="must search approved evidence",
    ):
        validate_evidence_discovery_result(
            requirement=build_requirement(),
            match=build_related_match(),
            called_tools=["list_verified_skills"],
            observed_evidence_ids=["EVD-DOCKER"],
            repository=build_repository(),
            user_id="USER-001",
        )


def test_agent_cannot_cite_unobserved_evidence() -> None:
    """The final match may cite only IDs returned by tools."""

    with pytest.raises(
        EvidenceDiscoveryValidationError,
        match="was not returned by its tools",
    ):
        validate_evidence_discovery_result(
            requirement=build_requirement(),
            match=build_related_match(),
            called_tools=["search_approved_evidence"],
            observed_evidence_ids=[],
            repository=build_repository(),
            user_id="USER-001",
        )


def test_agent_cannot_cite_other_users_evidence() -> None:
    """Observed output cannot bypass repository user isolation."""

    with pytest.raises(
        EvidenceDiscoveryValidationError,
        match="does not belong to the authenticated user",
    ):
        validate_evidence_discovery_result(
            requirement=build_requirement(),
            match=build_related_match(),
            called_tools=["search_approved_evidence"],
            observed_evidence_ids=["EVD-DOCKER"],
            repository=build_repository(),
            user_id="USER-002",
        )
