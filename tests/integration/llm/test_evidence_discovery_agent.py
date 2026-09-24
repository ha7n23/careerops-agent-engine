"""Live tests for the bounded batched LLM evidence agent."""

import json
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
def test_batched_docker_and_kubernetes_evidence_discovery() -> None:
    """One batch must distinguish direct from related evidence."""

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

    matches = runner.discover_for_requirements(
        [
            JobRequirement(
                requirement_id="REQ-DOCKER",
                name="Docker",
                category=RequirementCategory.ESSENTIAL,
                evidence_expected=(
                    "Experience containerising applications with Docker."
                ),
                importance_score=4,
                source_text=("Docker containerisation experience is required."),
            ),
            JobRequirement(
                requirement_id="REQ-KUBERNETES",
                name="Kubernetes",
                category=RequirementCategory.DESIRABLE,
                evidence_expected=("Exposure to Kubernetes container orchestration."),
                importance_score=2,
                source_text=("Exposure to Kubernetes is beneficial."),
            ),
        ],
        user_id="USER-INTEGRATION",
    )

    print(
        "Live batched evidence matches:\n"
        + json.dumps(
            [match.model_dump(mode="json") for match in matches],
            indent=2,
        )
    )

    assert [match.requirement_id for match in matches] == [
        "REQ-DOCKER",
        "REQ-KUBERNETES",
    ]

    docker_match, kubernetes_match = matches

    assert docker_match.match_strength is MatchStrength.STRONG
    assert docker_match.direct_evidence_ids == ["EVD-DOCKER"]
    assert docker_match.related_evidence_ids == []
    assert docker_match.gap is False

    assert kubernetes_match.match_strength is MatchStrength.RELATED
    assert kubernetes_match.direct_evidence_ids == []
    assert kubernetes_match.related_evidence_ids == ["EVD-DOCKER"]
    assert kubernetes_match.gap is True
