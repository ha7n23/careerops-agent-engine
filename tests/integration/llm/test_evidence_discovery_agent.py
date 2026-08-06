"""Live tests for the bounded Gemini evidence agent."""

import os

import pytest

from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.llm.factory import (
    create_evidence_discovery_runner,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Live LLM tests are disabled.",
)
def test_docker_is_related_not_direct_kubernetes_evidence() -> None:
    """Gemini must preserve the Docker/Kubernetes distinction."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-INTEGRATION": [
                CareerEvidence(
                    evidence_id="EVD-DOCKER",
                    category=EvidenceCategory.PROJECT,
                    title="Containerised FastAPI Service",
                    verification_status=(VerificationStatus.APPROVED),
                    technologies=[
                        "Python",
                        "FastAPI",
                        "Docker",
                    ],
                    capabilities=[
                        "Application containerisation",
                    ],
                    approved_claims=[
                        "Containerised a FastAPI application using Docker."
                    ],
                    source_references=[
                        SourceReference(
                            source_type=(EvidenceSourceType.MANUAL_ENTRY),
                            source_id="SRC-DOCKER",
                        )
                    ],
                )
            ]
        }
    )

    runner = create_evidence_discovery_runner(repository)

    match = runner.discover(
        JobRequirement(
            requirement_id="REQ-KUBERNETES",
            name="Kubernetes",
            category=RequirementCategory.DESIRABLE,
            evidence_expected=("Exposure to Kubernetes container orchestration."),
            importance_score=2,
            source_text=("Exposure to Kubernetes is beneficial."),
        ),
        user_id="USER-INTEGRATION",
    )

    assert match.match_strength is MatchStrength.RELATED
    assert match.direct_evidence_ids == []
    assert match.related_evidence_ids == ["EVD-DOCKER"]
    assert match.gap is True
