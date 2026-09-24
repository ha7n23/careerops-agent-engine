"""Tests for read-only Evidence Registry access."""

import pytest

from careerops_agent_engine.application.exceptions import (
    CareerEvidenceUnavailableError,
)
from careerops_agent_engine.application.services.evidence_registry import (
    EvidenceRegistryService,
)
from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_evidence(
    evidence_id: str,
    *,
    title: str,
) -> CareerEvidence:
    """Create one approved evidence record."""

    return CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title=title,
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python", "FastAPI"],
        capabilities=["API development"],
        approved_claims=[f"Built {title} using Python and FastAPI."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id=f"SRC-{evidence_id}",
            )
        ],
    )


def build_service() -> EvidenceRegistryService:
    """Create isolated evidence for two users."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence(
                    "EVD-001",
                    title="CareerOps",
                ),
                build_evidence(
                    "EVD-002",
                    title="Analytics API",
                ),
            ],
            "USER-002": [
                build_evidence(
                    "EVD-OTHER",
                    title="Private Project",
                )
            ],
        }
    )

    return EvidenceRegistryService(repository)


def test_list_returns_only_authenticated_users_evidence() -> None:
    """Registry listing must remain inside the user boundary."""

    evidence = build_service().list_approved(
        user_id="USER-001",
        limit=100,
    )

    assert [item.evidence_id for item in evidence] == [
        "EVD-001",
        "EVD-002",
    ]


def test_get_returns_one_approved_record() -> None:
    """A user may retrieve their own approved evidence."""

    evidence = build_service().get_approved(
        user_id="USER-001",
        evidence_id="EVD-001",
    )

    assert evidence.evidence_id == "EVD-001"
    assert evidence.title == "CareerOps"


def test_get_hides_another_users_record() -> None:
    """Cross-user access must use the same opaque unavailable error."""

    with pytest.raises(
        CareerEvidenceUnavailableError,
        match="approved evidence record is unavailable",
    ):
        build_service().get_approved(
            user_id="USER-001",
            evidence_id="EVD-OTHER",
        )


def test_get_hides_unknown_record() -> None:
    """Unknown identifiers must not be distinguishable from cross-user IDs."""

    with pytest.raises(
        CareerEvidenceUnavailableError,
        match="approved evidence record is unavailable",
    ):
        build_service().get_approved(
            user_id="USER-001",
            evidence_id="EVD-MISSING",
        )
