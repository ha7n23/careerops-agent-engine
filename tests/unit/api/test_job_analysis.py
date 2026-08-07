"""Tests for the complete evidence-grounded job-analysis API."""

from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_job_analysis_service,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
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
from careerops_agent_engine.main import app


class FakeRequirementExtractor:
    """Deterministic requirement extraction."""

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
                    evidence_expected=("Practical Python engineering experience."),
                    importance_score=5,
                    source_text=("Strong Python experience is required."),
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
    """Return controlled evidence matches."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return one strong and one missing match."""

        assert user_id == "USER-API-001"

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
    """Return one deterministic Python proposal."""

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Generate a grounded proposal."""

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
    """Return a deterministic supported report."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Verify the test proposal."""

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
    """Create approved evidence for the API test user."""

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
            "USER-API-001": [evidence],
        }
    )


def override_job_analysis_service() -> JobAnalysisService:
    """Return a service backed by deterministic fakes."""

    repository = build_repository()

    return JobAnalysisService(
        requirement_extractor=FakeRequirementExtractor(),
        evidence_discovery_runner=FakeEvidenceDiscoveryRunner(),
        cv_proposal_service=CVProposalGenerationService(
            repository=repository,
            generator=FakeCVProposalGenerator(),
        ),
        cv_claim_verification_service=(
            CVClaimVerificationService(
                repository=repository,
                verifier=FakeClaimVerifier(),
            )
        ),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide an isolated API client."""

    app.dependency_overrides[get_job_analysis_service] = override_job_analysis_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_job_analysis_returns_verified_cv_proposal(
    client: TestClient,
) -> None:
    """A valid request should expose a reviewable proposal."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "job_id": "JOB-API-001",
            "job_description": (
                "We require strong Python and LangGraph "
                "workflow development experience."
            ),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["job_id"] == "JOB-API-001"
    assert body["role_title"] == "Junior AI Engineer"
    assert body["fit_score"] == 55.56

    assert len(body["requirements"]) == 2
    assert len(body["evidence_matches"]) == 2

    assert len(body["cv_proposals"]) == 1
    assert len(body["claim_verification_reports"]) == 1

    proposal = body["cv_proposals"][0]

    assert proposal["requirement_ids"] == ["REQ-PYTHON"]
    assert proposal["supporting_evidence_ids"] == ["EVD-PYTHON"]
    assert proposal["requires_human_approval"] is True

    proposal_id = proposal["proposal_id"]

    assert body["reviewable_proposal_ids"] == [proposal_id]
    assert body["blocked_proposal_ids"] == []

    report = body["claim_verification_reports"][0]

    assert report["proposal_id"] == proposal_id
    assert report["fully_supported"] is True
    assert report["unsupported_claims"] == []

    assert [event["event"] for event in body["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "evidence_discovery_completed",
        "fit_score_calculated",
        "cv_proposals_generated",
        "cv_proposals_verified",
        "job_analysis_completed",
    ]


def test_short_job_description_returns_422(
    client: TestClient,
) -> None:
    """Invalid workflow input should become an API error."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "job_id": "JOB-API-002",
            "job_description": "Too short",
        },
    )

    assert response.status_code == 422

    assert "at least 20 characters" in (response.json()["detail"])


def test_missing_user_header_returns_401(
    client: TestClient,
) -> None:
    """Requests require trusted user context."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-003",
            "job_description": ("We require strong Python development experience."),
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == ("The X-User-ID header is required.")


def test_client_cannot_submit_match_or_proposal_state(
    client: TestClient,
) -> None:
    """Clients cannot control internal analysis state."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "job_id": "JOB-API-004",
            "job_description": ("We require strong Python development experience."),
            "matched_requirement_ids": ["REQ-PYTHON"],
            "reviewable_proposal_ids": ["CVP-INVENTED"],
        },
    )

    assert response.status_code == 422
