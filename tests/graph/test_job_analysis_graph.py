"""Tests for the evidence-grounded job-analysis graph."""

from collections.abc import Sequence

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
    VerificationStatus,
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
    """Deterministic replacement for requirement extraction."""

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Return predictable requirements."""

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
                    evidence_expected=("Implementation of stateful agent workflows."),
                    importance_score=4,
                    source_text=("Experience with LangGraph is required."),
                ),
            ],
        )


class FakeEvidenceDiscoveryRunner:
    """Return deterministic evidence matches."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return controlled match results."""

        assert user_id == "USER-TEST-001"

        if requirement.requirement_id == "REQ-PYTHON":
            return EvidenceMatch(
                requirement_id=requirement.requirement_id,
                match_strength=MatchStrength.STRONG,
                direct_evidence_ids=["EVD-PYTHON"],
                related_evidence_ids=[],
                explanation=(
                    "Approved evidence directly demonstrates Python experience."
                ),
                gap=False,
            )

        return EvidenceMatch(
            requirement_id=requirement.requirement_id,
            match_strength=MatchStrength.NONE,
            direct_evidence_ids=[],
            related_evidence_ids=[],
            explanation="No approved LangGraph evidence was found.",
            gap=True,
        )


class FakeCVProposalGenerator:
    """Generate one deterministic grounded CV proposal."""

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Return a proposal using supplied approved evidence."""

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
    """Mark the deterministic proposal as fully supported."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return a fully supported verification report."""

        assert approved_evidence

        return CVClaimVerificationReport(
            proposal_id=proposal.proposal_id,
            claims=[
                ClaimAssessment(
                    claim_text=proposal.proposed_text,
                    supported=True,
                    supporting_evidence_ids=[approved_evidence[0].evidence_id],
                    explanation=(
                        "The approved evidence directly supports the proposal."
                    ),
                )
            ],
            coverage_complete=True,
            coverage_notes=[],
            fully_supported=True,
            unsupported_claims=[],
        )


def build_repository() -> InMemoryEvidenceRepository:
    """Create approved evidence for the test user."""

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


def build_test_graph():
    """Create a graph backed entirely by deterministic adapters."""

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
    )


def test_valid_job_runs_complete_grounded_analysis_path() -> None:
    """Valid input should reach proposal verification."""

    graph = build_test_graph()

    result = graph.invoke(
        {
            "job_id": "JOB-001",
            "user_id": "USER-TEST-001",
            "job_description": (
                "We require strong Python development and "
                "LangGraph workflow experience."
            ),
            "audit_events": [],
        }
    )

    assert result["status"] == "completed"
    assert result["validation_error"] is None
    assert result["role_title"] == "Junior AI Engineer"

    assert len(result["requirements"]) == 2
    assert len(result["evidence_matches"]) == 2

    # Python weight = 5, LangGraph weight = 4.
    assert result["fit_score"] == 55.56

    # Only the strong Python match may produce a CV proposal.
    assert len(result["cv_proposals"]) == 1
    assert len(result["claim_verification_reports"]) == 1

    proposal = CVChangeProposal.model_validate(result["cv_proposals"][0])

    assert proposal.requirement_ids == ["REQ-PYTHON"]
    assert proposal.supporting_evidence_ids == ["EVD-PYTHON"]
    assert proposal.requires_human_approval is True

    report = CVClaimVerificationReport.model_validate(
        result["claim_verification_reports"][0]
    )

    assert report.proposal_id == proposal.proposal_id
    assert report.fully_supported is True

    assert result["reviewable_proposal_ids"] == [proposal.proposal_id]
    assert result["blocked_proposal_ids"] == []

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "evidence_discovery_completed",
        "fit_score_calculated",
        "cv_proposals_generated",
        "cv_proposals_verified",
        "job_analysis_completed",
    ]


def test_invalid_job_skips_downstream_workflow() -> None:
    """Invalid input should route directly to rejection."""

    graph = build_test_graph()

    result = graph.invoke(
        {
            "job_id": "JOB-002",
            "user_id": "USER-TEST-001",
            "job_description": "Too short",
            "audit_events": [],
        }
    )

    assert result["status"] == "invalid"
    assert result["validation_error"] is not None

    assert "requirements" not in result
    assert "evidence_matches" not in result
    assert "fit_score" not in result
    assert "cv_proposals" not in result
    assert "claim_verification_reports" not in result

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_invalid",
        "job_analysis_rejected",
    ]
