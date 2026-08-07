"""Tests for the durable CareerOps job-analysis API."""

from collections.abc import Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    nullcontext,
)
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

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
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
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


class SharedCheckpointerFactory:
    """Keep one in-memory checkpoint store across API requests."""

    def __init__(self) -> None:
        """Create one shared test checkpointer."""

        self.checkpointer = InMemorySaver()

    def __call__(
        self,
    ) -> AbstractContextManager[BaseCheckpointSaver[str]]:
        """Return the same saver without closing or replacing it."""

        saver = cast(
            BaseCheckpointSaver[str],
            self.checkpointer,
        )

        return nullcontext(saver)


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
    """Return one strong and one missing match."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return deterministic evidence matching."""

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
    """Generate one grounded proposal."""

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Return deterministic CV wording."""

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
    """Verify the generated proposal."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return a fully supported report."""

        return CVClaimVerificationReport(
            proposal_id=proposal.proposal_id,
            claims=[
                ClaimAssessment(
                    claim_text=proposal.proposed_text,
                    supported=True,
                    supporting_evidence_ids=[approved_evidence[0].evidence_id],
                    explanation=("Approved evidence directly supports the proposal."),
                )
            ],
            coverage_complete=True,
            coverage_notes=[],
            fully_supported=True,
            unsupported_claims=[],
        )


def build_repository() -> InMemoryEvidenceRepository:
    """Create approved evidence for the API user."""

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


def build_test_service() -> JobAnalysisService:
    """Create one durable service for the whole API test."""

    repository = build_repository()

    return JobAnalysisService(
        requirement_extractor=FakeRequirementExtractor(),
        evidence_discovery_runner=(FakeEvidenceDiscoveryRunner()),
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
        checkpointer_factory=SharedCheckpointerFactory(),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a client whose checkpoints survive requests."""

    service = build_test_service()

    def override_service() -> JobAnalysisService:
        return service

    app.dependency_overrides[get_job_analysis_service] = override_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def start_reviewable_analysis(
    client: TestClient,
    *,
    job_id: str,
) -> dict[str, Any]:
    """Start a workflow expected to pause for review."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "job_id": job_id,
            "job_description": (
                "We require strong Python and LangGraph "
                "workflow development experience."
            ),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "awaiting_review"

    return body


def test_job_analysis_pauses_then_approves(
    client: TestClient,
) -> None:
    """A verified proposal should pause and later complete."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-001",
    )

    thread_id = paused["thread_id"]

    assert isinstance(thread_id, str)
    assert thread_id.startswith("THR-")

    assert paused["fit_score"] == 55.56
    assert len(paused["cv_proposals"]) == 1
    assert len(paused["claim_verification_reports"]) == 1

    proposal = paused["cv_proposals"][0]
    proposal_id = proposal["proposal_id"]

    assert paused["reviewable_proposal_ids"] == [proposal_id]
    assert paused["blocked_proposal_ids"] == []

    assert paused["review"]["type"] == ("cv_proposal_review")
    assert paused["review"]["allowed_actions"] == [
        "approve",
        "reject",
    ]

    assert [event["event"] for event in paused["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "evidence_discovery_completed",
        "fit_score_calculated",
        "cv_proposals_generated",
        "cv_proposals_verified",
    ]

    response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "approve",
            "approved_proposal_ids": [proposal_id],
            "rejected_proposal_ids": [],
            "edits": [],
            "reviewer_comment": None,
        },
    )

    assert response.status_code == 200

    completed = response.json()

    assert completed["status"] == "completed"
    assert completed["thread_id"] == thread_id
    assert completed["review_status"] == "approved"

    assert len(completed["final_cv_proposals"]) == 1

    assert completed["final_cv_proposals"][0]["proposal_id"] == proposal_id

    assert [event["event"] for event in completed["audit_events"]] == [
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


def test_job_analysis_can_be_rejected(
    client: TestClient,
) -> None:
    """Human rejection should produce no final proposal."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-REJECT",
    )

    thread_id = paused["thread_id"]
    proposal_id = paused["cv_proposals"][0]["proposal_id"]

    response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "reject",
            "approved_proposal_ids": [],
            "rejected_proposal_ids": [proposal_id],
            "edits": [],
            "reviewer_comment": ("Do not use this proposal."),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["review_status"] == "rejected"
    assert body["final_cv_proposals"] == []


def test_wrong_proposal_id_is_rejected(
    client: TestClient,
) -> None:
    """A reviewer cannot approve an unrelated proposal ID."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-WRONG-ID",
    )

    thread_id = paused["thread_id"]

    response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "approve",
            "approved_proposal_ids": ["CVP-INVENTED"],
            "rejected_proposal_ids": [],
            "edits": [],
            "reviewer_comment": None,
        },
    )

    assert response.status_code == 422

    assert "approve every reviewable proposal" in (response.json()["detail"])


def test_other_user_cannot_resume_thread(
    client: TestClient,
) -> None:
    """A durable thread must remain bound to its owner."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-CROSS-USER",
    )

    thread_id = paused["thread_id"]
    proposal_id = paused["cv_proposals"][0]["proposal_id"]

    response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-OTHER",
        },
        json={
            "action": "approve",
            "approved_proposal_ids": [proposal_id],
            "rejected_proposal_ids": [],
            "edits": [],
            "reviewer_comment": None,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("The requested review thread is unavailable.")


def test_short_job_description_returns_422(
    client: TestClient,
) -> None:
    """Invalid input should still return 422."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "job_id": "JOB-API-SHORT",
            "job_description": "Too short",
        },
    )

    assert response.status_code == 422


def test_missing_user_header_returns_401(
    client: TestClient,
) -> None:
    """Starting a workflow requires user context."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-NO-USER",
            "job_description": ("Strong Python development experience is required."),
        },
    )

    assert response.status_code == 401


def test_client_cannot_submit_internal_workflow_state(
    client: TestClient,
) -> None:
    """Internal workflow state cannot be injected by clients."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "job_id": "JOB-API-INTERNAL",
            "job_description": ("Strong Python development experience is required."),
            "reviewable_proposal_ids": ["CVP-INVENTED"],
        },
    )

    assert response.status_code == 422
