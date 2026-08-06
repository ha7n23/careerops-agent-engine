"""Tests for the in-memory evidence repository."""

import pytest

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
    *,
    evidence_id: str,
    title: str,
    category: EvidenceCategory,
    status: VerificationStatus,
    technologies: list[str],
    claims: list[str],
) -> CareerEvidence:
    """Create one evidence record for repository tests."""

    return CareerEvidence(
        evidence_id=evidence_id,
        category=category,
        title=title,
        verification_status=status,
        technologies=technologies,
        capabilities=[],
        approved_claims=claims,
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id=f"SRC-{evidence_id}",
            )
        ],
    )


@pytest.fixture
def repository() -> InMemoryEvidenceRepository:
    """Create isolated evidence for two different users."""

    return InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence(
                    evidence_id="EVD-LANGGRAPH",
                    title="CareerOps Agent Engine",
                    category=EvidenceCategory.PROJECT,
                    status=VerificationStatus.APPROVED,
                    technologies=[
                        "Python",
                        "LangGraph",
                        "Docker",
                    ],
                    claims=[
                        "Built a stateful LangGraph workflow.",
                    ],
                ),
                build_evidence(
                    evidence_id="EVD-PYTHON",
                    title="Verified Python Skill",
                    category=EvidenceCategory.SKILL,
                    status=VerificationStatus.APPROVED,
                    technologies=["Python"],
                    claims=[
                        "Used Python across multiple software projects.",
                    ],
                ),
                build_evidence(
                    evidence_id="EVD-KUBERNETES-REJECTED",
                    title="Unverified Kubernetes Claim",
                    category=EvidenceCategory.SKILL,
                    status=VerificationStatus.REJECTED,
                    technologies=["Kubernetes"],
                    claims=[
                        "Used Kubernetes in production.",
                    ],
                ),
            ],
            "USER-002": [
                build_evidence(
                    evidence_id="EVD-KUBERNETES",
                    title="Kubernetes Deployment",
                    category=EvidenceCategory.PROJECT,
                    status=VerificationStatus.APPROVED,
                    technologies=["Kubernetes"],
                    claims=[
                        "Deployed an application to Kubernetes.",
                    ],
                )
            ],
        }
    )


def test_search_returns_relevant_approved_evidence(
    repository: InMemoryEvidenceRepository,
) -> None:
    """Approved matching evidence should be returned."""

    results = repository.search_approved(
        user_id="USER-001",
        query="LangGraph stateful workflow",
    )

    assert [result.evidence_id for result in results] == ["EVD-LANGGRAPH"]


def test_search_excludes_rejected_and_cross_user_evidence(
    repository: InMemoryEvidenceRepository,
) -> None:
    """Search must preserve approval and user boundaries."""

    results = repository.search_approved(
        user_id="USER-001",
        query="Kubernetes",
    )

    assert results == []


def test_get_approved_rejects_unapproved_record(
    repository: InMemoryEvidenceRepository,
) -> None:
    """A rejected record must not be retrievable as approved."""

    result = repository.get_approved(
        user_id="USER-001",
        evidence_id="EVD-KUBERNETES-REJECTED",
    )

    assert result is None


def test_list_returns_only_approved_user_records(
    repository: InMemoryEvidenceRepository,
) -> None:
    """Listing must remain inside one verified evidence set."""

    results = repository.list_approved(
        user_id="USER-001",
    )

    assert {result.evidence_id for result in results} == {
        "EVD-LANGGRAPH",
        "EVD-PYTHON",
    }


def test_duplicate_user_evidence_ids_are_rejected() -> None:
    """Duplicate identifiers would make retrieval ambiguous."""

    evidence = build_evidence(
        evidence_id="EVD-001",
        title="Example",
        category=EvidenceCategory.PROJECT,
        status=VerificationStatus.APPROVED,
        technologies=["Python"],
        claims=["Built a Python project."],
    )

    with pytest.raises(
        ValueError,
        match="must be unique within a user",
    ):
        InMemoryEvidenceRepository(
            {
                "USER-001": [
                    evidence,
                    evidence,
                ]
            }
        )
