"""Synthetic approved evidence used during local development."""

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

DEVELOPMENT_USER_ID = "USER-DEMO-001"


def create_development_evidence_repository() -> InMemoryEvidenceRepository:
    """Create a repository containing synthetic approved evidence."""

    careerops_evidence = CareerEvidence(
        evidence_id="EVD-DEMO-CAREEROPS",
        category=EvidenceCategory.PROJECT,
        title="CareerOps Agent Engine",
        verification_status=VerificationStatus.APPROVED,
        technologies=[
            "Python",
            "FastAPI",
            "LangChain",
            "LangGraph",
            "LangSmith",
            "Docker",
        ],
        capabilities=[
            "Stateful agent workflow development",
            "Structured LLM output",
            "Human-in-the-loop architecture",
            "LLM tracing and evaluation",
            "Application containerisation",
        ],
        approved_claims=[
            ("Built a stateful LangGraph workflow with structured Gemini extraction."),
            ("Integrated LangSmith tracing for model and agent observability."),
            ("Exposed the workflow through a tested FastAPI service."),
            ("Containerised Python API applications using Docker."),
        ],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id="SRC-DEMO-CAREEROPS",
            )
        ],
    )

    aws_evidence = CareerEvidence(
        evidence_id="EVD-DEMO-AWS",
        category=EvidenceCategory.PROJECT,
        title="AWS-Deployed AI API",
        verification_status=VerificationStatus.APPROVED,
        technologies=[
            "Python",
            "FastAPI",
            "Docker",
            "AWS",
        ],
        capabilities=[
            "Cloud application deployment",
            "Container deployment",
        ],
        approved_claims=[("Deployed a containerised FastAPI application to AWS.")],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id="SRC-DEMO-AWS",
            )
        ],
    )

    return InMemoryEvidenceRepository(
        {
            DEVELOPMENT_USER_ID: [
                careerops_evidence,
                aws_evidence,
            ]
        }
    )
