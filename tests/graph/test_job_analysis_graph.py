"""Tests for the durable evidence-grounded job-analysis graph."""

from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from careerops_agent_engine.agents.graphs.job_analysis import (
    build_job_analysis_graph,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    ReviewAction,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)
from careerops_agent_engine.domain.models.cv import CVChangeProposal
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)
from careerops_agent_engine.domain.models.verification import (
    ClaimAssessment,
    CVClaimVerificationReport,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


class FakeRequirementExtractor:
    """Deterministic requirement extractor."""

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Return predictable test requirements."""

        del job_description, job_id

        return JobRequirementExtraction(
            role_title="Junior AI Engineer",
            requirements=[
                JobRequirement(
                    requirement_id="REQ-PYTHON",
                    name="Python",
                    category=RequirementCategory.ESSENTIAL,
                    evidence_expected=(
                        "Practical Python software-engineering experience."
                    ),
                    importance_score=5,
                    source_text=("Strong Python development experience is required."),
                ),
                JobRequirement(
                    requirement_id="REQ-LANGGRAPH",
                    name="LangGraph",
                    category=RequirementCategory.ESSENTIAL,
                    evidence_expected=("Stateful agent workflow experience."),
                    importance_score=4,
                    source_text=("LangGraph experience is required."),
                ),
            ],
        )


class FakeEvidenceDiscoveryRunner:
    """Return one strong match and one genuine gap."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return deterministic evidence discovery."""

        assert user_id == "USER-TEST-001"

        if requirement.requirement_id == "REQ-PYTHON":
            return EvidenceMatch(
                requirement_id=requirement.requirement_id,
                match_strength=MatchStrength.STRONG,
                direct_evidence_ids=["EVD-PYTHON"],
                related_evidence_ids=[],
                explanation=("Approved evidence directly supports Python."),
                gap=False,
            )

        return EvidenceMatch(
            requirement_id=requirement.requirement_id,
            match_strength=MatchStrength.NONE,
            direct_evidence_ids=[],
            related_evidence_ids=[],
            explanation="No approved LangGraph evidence exists.",
            gap=True,
        )


class FakeCVProposalGenerator:
    """Return one predictable Python CV proposal."""

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Generate a grounded test proposal."""

        del job_id, evidence_match

        return CVChangeProposal(
            proposal_id=proposal_id,
            section=CVSection.PROJECTS,
            proposed_text=("Built a FastAPI application using Python."),
            requirement_ids=[requirement.requirement_id],
            supporting_evidence_ids=[
                evidence.evidence_id for evidence in approved_evidence
            ],
            confidence_score=0.95,
            warnings=[],
        )


class FakeClaimVerifier:
    """Mark the generated Python proposal as supported."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return a complete supported report."""

        assert approved_evidence

        return CVClaimVerificationReport(
            proposal_id=proposal.proposal_id,
            claims=[
                ClaimAssessment(
                    claim_text=proposal.proposed_text,
                    supported=True,
                    supporting_evidence_ids=[approved_evidence[0].evidence_id],
                    explanation=(
                        "The approved project evidence directly supports the proposal."
                    ),
                )
            ],
            coverage_complete=True,
            coverage_notes=[],
            fully_supported=True,
            unsupported_claims=[],
        )


def build_repository() -> InMemoryEvidenceRepository:
    """Create approved evidence for the graph test user."""

    evidence = CareerEvidence(
        evidence_id="EVD-PYTHON",
        category=EvidenceCategory.PROJECT,
        title="Python API Project",
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python", "FastAPI"],
        capabilities=["API development"],
        approved_claims=["Built a FastAPI application using Python."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id="SRC-PYTHON",
            )
        ],
    )

    return InMemoryEvidenceRepository(
        {
            "USER-TEST-001": [evidence],
        }
    )


def build_test_graph(
    checkpointer: InMemorySaver,
):
    """Create a durable graph backed by deterministic adapters."""

    repository = build_repository()

    proposal_service = CVProposalGenerationService(
        repository=repository,
        generator=FakeCVProposalGenerator(),
    )

    verification_service = CVClaimVerificationService(
        repository=repository,
        verifier=FakeClaimVerifier(),
    )

    return build_job_analysis_graph(
        requirement_extractor=FakeRequirementExtractor(),
        evidence_discovery_runner=FakeEvidenceDiscoveryRunner(),
        cv_proposal_service=proposal_service,
        cv_claim_verification_service=verification_service,
        checkpointer=checkpointer,
    )


def test_valid_job_pauses_then_resumes_with_approval() -> None:
    """A verified proposal should require durable human approval."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "THREAD-GRAPH-APPROVE",
        }
    }

    paused = graph.invoke(
        {
            "job_id": "JOB-001",
            "user_id": "USER-TEST-001",
            "job_description": (
                "We require strong Python development and "
                "LangGraph workflow experience."
            ),
            "audit_events": [],
        },
        config=config,
    )

    assert "__interrupt__" in paused

    proposal = CVChangeProposal.model_validate(paused["cv_proposals"][0])

    assert paused["fit_score"] == 55.56
    assert paused["reviewable_proposal_ids"] == [proposal.proposal_id]
    assert paused["blocked_proposal_ids"] == []

    interrupts = paused["__interrupt__"]

    assert len(interrupts) == 1

    payload = interrupts[0].value

    assert payload["type"] == "cv_proposal_review"
    assert payload["allowed_actions"] == [
        "approve",
        "reject",
    ]

    decision = CVReviewDecision(
        action=ReviewAction.APPROVE,
        approved_proposal_ids=[proposal.proposal_id],
    )

    resumed = graph.invoke(
        Command(resume=decision.model_dump(mode="json")),
        config=config,
    )

    assert resumed["status"] == "completed"
    assert resumed["review_status"] == "approved"

    assert len(resumed["final_cv_proposals"]) == 1

    final_proposal = CVChangeProposal.model_validate(resumed["final_cv_proposals"][0])

    assert final_proposal.proposal_id == proposal.proposal_id

    assert [event["event"] for event in resumed["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "evidence_discovery_completed",
        "fit_score_calculated",
        "cv_proposals_generated",
        "cv_proposals_verified",
        "human_review_received",
        "human_review_finalized",
        "job_analysis_completed",
    ]


def test_valid_job_can_be_rejected_after_interrupt() -> None:
    """Human rejection should complete with no final proposal."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "THREAD-GRAPH-REJECT",
        }
    }

    paused = graph.invoke(
        {
            "job_id": "JOB-002",
            "user_id": "USER-TEST-001",
            "job_description": (
                "We require strong Python development and "
                "LangGraph workflow experience."
            ),
            "audit_events": [],
        },
        config=config,
    )

    proposal = CVChangeProposal.model_validate(paused["cv_proposals"][0])

    decision = CVReviewDecision(
        action=ReviewAction.REJECT,
        rejected_proposal_ids=[proposal.proposal_id],
    )

    resumed = graph.invoke(
        Command(resume=decision.model_dump(mode="json")),
        config=config,
    )

    assert resumed["status"] == "completed"
    assert resumed["review_status"] == "rejected"
    assert resumed["final_cv_proposals"] == []


def test_invalid_job_skips_human_review() -> None:
    """Invalid input should terminate without an interrupt."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "THREAD-GRAPH-INVALID",
        }
    }

    result = graph.invoke(
        {
            "job_id": "JOB-003",
            "user_id": "USER-TEST-001",
            "job_description": "Too short",
            "audit_events": [],
        },
        config=config,
    )

    assert result["status"] == "invalid"
    assert "__interrupt__" not in result
    assert "cv_proposals" not in result

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_invalid",
        "job_analysis_rejected",
    ]
