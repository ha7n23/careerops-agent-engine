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
from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerificationRequest,
)
from careerops_agent_engine.application.ports.cv_proposal_generator import (
    CVProposalGenerationRequest,
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
from careerops_agent_engine.domain.models.audit import (
    CVReviewAuditEntry,
    JobAnalysisRunSnapshot,
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


class InMemoryJobAnalysisAuditRepository:
    """Store business audit records in memory for API tests."""

    def __init__(self) -> None:
        """Create empty run and review stores."""

        self._runs: dict[
            str,
            JobAnalysisRunSnapshot,
        ] = {}

        self._reviews: dict[
            str,
            list[CVReviewAuditEntry],
        ] = {}

    def save_run(
        self,
        snapshot: JobAnalysisRunSnapshot,
    ) -> None:
        """Create or replace the latest run snapshot."""

        self._runs[snapshot.thread_id] = snapshot.model_copy(deep=True)

    def save_review_result(
        self,
        *,
        snapshot: JobAnalysisRunSnapshot,
        review: CVReviewAuditEntry,
    ) -> None:
        """Persist a run update and one review event."""

        self.save_run(snapshot)

        reviews = self._reviews.setdefault(
            review.thread_id,
            [],
        )

        reviews.append(review.model_copy(deep=True))

    def get_run(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> JobAnalysisRunSnapshot | None:
        """Return a user-scoped run."""

        snapshot = self._runs.get(thread_id)

        if snapshot is None or snapshot.user_id != user_id:
            return None

        return snapshot.model_copy(deep=True)

    def list_reviews(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[CVReviewAuditEntry]:
        """Return ordered user-scoped review history."""

        snapshot = self._runs.get(thread_id)

        if snapshot is None or snapshot.user_id != user_id:
            return []

        reviews = self._reviews.get(
            thread_id,
            [],
        )

        return [
            review.model_copy(deep=True)
            for review in sorted(
                reviews,
                key=lambda item: item.sequence_number,
            )
        ]


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

    def discover_for_requirements(
        self,
        requirements: Sequence[JobRequirement],
        *,
        user_id: str,
    ) -> list[EvidenceMatch]:
        """Return deterministic matches for the requirement set."""

        return [
            self.discover(
                requirement,
                user_id=user_id,
            )
            for requirement in requirements
        ]


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

    def generate_batch(
        self,
        *,
        job_id: str,
        requests: Sequence[CVProposalGenerationRequest],
    ) -> list[CVChangeProposal]:
        """Return deterministic proposals for one simulated batch call."""

        return [
            self.generate(
                proposal_id=request.proposal_id,
                job_id=job_id,
                requirement=request.requirement,
                evidence_match=request.evidence_match,
                approved_evidence=request.approved_evidence,
            )
            for request in requests
        ]

    def regenerate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
        previous_proposal: CVChangeProposal,
        reviewer_feedback: str,
    ) -> CVChangeProposal:
        """Return controlled regenerated API-test wording."""

        del (
            job_id,
            evidence_match,
            previous_proposal,
        )

        if "kubernetes" in reviewer_feedback.casefold():
            proposed_text = "Built a Kubernetes-based Python FastAPI application."
        else:
            proposed_text = "Built a concise Python API using FastAPI."

        return CVChangeProposal(
            proposal_id=proposal_id,
            section=CVSection.PROJECTS,
            proposed_text=proposed_text,
            requirement_ids=[requirement.requirement_id],
            supporting_evidence_ids=[
                evidence.evidence_id for evidence in approved_evidence
            ],
            confidence_score=0.95,
            warnings=[],
        )


class FakeClaimVerifier:
    """Verify grounded text and reject an invented latency metric."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return verification based on current proposal wording."""

        lowered_text = proposal.proposed_text.casefold()

        if "70" in lowered_text and "latency" in lowered_text:
            unsupported_claim = "Reduced production latency by 70 percent."

            return CVClaimVerificationReport(
                proposal_id=proposal.proposal_id,
                claims=[
                    ClaimAssessment(
                        claim_text=unsupported_claim,
                        supported=False,
                        supporting_evidence_ids=[],
                        explanation=(
                            "No approved evidence supports this latency metric."
                        ),
                    )
                ],
                coverage_complete=True,
                coverage_notes=[],
                fully_supported=False,
                unsupported_claims=[unsupported_claim],
            )

        if "kubernetes" in lowered_text:
            unsupported_claim = "Built a Kubernetes-based Python FastAPI application."

            return CVClaimVerificationReport(
                proposal_id=proposal.proposal_id,
                claims=[
                    ClaimAssessment(
                        claim_text=unsupported_claim,
                        supported=False,
                        supporting_evidence_ids=[],
                        explanation=(
                            "Approved evidence does not support Kubernetes experience."
                        ),
                    )
                ],
                coverage_complete=True,
                coverage_notes=[],
                fully_supported=False,
                unsupported_claims=[unsupported_claim],
            )

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

    def verify_batch(
        self,
        *,
        requests: Sequence[CVClaimVerificationRequest],
    ) -> list[CVClaimVerificationReport]:
        """Return deterministic reports for one simulated batch call."""

        return [
            self.verify(
                proposal=request.proposal,
                approved_evidence=request.approved_evidence,
            )
            for request in requests
        ]


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


def build_test_service(
    audit_repository: (InMemoryJobAnalysisAuditRepository),
) -> JobAnalysisService:
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
        audit_repository=audit_repository,
    )


@pytest.fixture
def audit_repository() -> InMemoryJobAnalysisAuditRepository:
    """Provide isolated business audit storage."""

    return InMemoryJobAnalysisAuditRepository()


@pytest.fixture
def client(
    audit_repository: (InMemoryJobAnalysisAuditRepository),
) -> Iterator[TestClient]:
    """Provide a client whose checkpoints survive requests."""

    service = build_test_service(audit_repository)

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


def test_initial_review_pause_is_persisted(
    client: TestClient,
    audit_repository: (InMemoryJobAnalysisAuditRepository),
) -> None:
    """Starting analysis should persist its business snapshot."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-AUDIT-INITIAL",
    )

    thread_id = paused["thread_id"]

    stored = audit_repository.get_run(
        user_id="USER-API-001",
        thread_id=thread_id,
    )

    assert stored is not None
    assert stored.thread_id == thread_id
    assert stored.job_id == "JOB-API-AUDIT-INITIAL"
    assert stored.status.value == "awaiting_review"
    assert len(stored.cv_proposals) == 1
    assert len(stored.claim_verification_reports) == 1

    # Business history must enforce the same user boundary.
    assert (
        audit_repository.get_run(
            user_id="USER-OTHER",
            thread_id=thread_id,
        )
        is None
    )

    assert (
        audit_repository.list_reviews(
            user_id="USER-API-001",
            thread_id=thread_id,
        )
        == []
    )


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
        "edit",
        "regenerate",
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

    assert "outside the current review set" in (response.json()["detail"])


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


def test_unsafe_human_edit_is_blocked_then_corrected(
    client: TestClient,
) -> None:
    """Unsafe human wording should pause again until corrected."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-EDIT-REWORK",
    )

    thread_id = paused["thread_id"]
    proposal_id = paused["cv_proposals"][0]["proposal_id"]

    unsafe_response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "edit",
            "approved_proposal_ids": [],
            "rejected_proposal_ids": [],
            "edits": [
                {
                    "proposal_id": proposal_id,
                    "edited_text": (
                        "Built a Python FastAPI application "
                        "and reduced production latency by "
                        "70 percent."
                    ),
                }
            ],
            "reviewer_comment": ("Testing unsupported human wording."),
        },
    )

    assert unsafe_response.status_code == 200

    rework = unsafe_response.json()

    assert rework["status"] == "awaiting_review"

    assert rework["review"]["allowed_actions"] == [
        "edit",
        "regenerate",
        "reject",
    ]

    assert rework["reviewable_proposal_ids"] == []
    assert proposal_id in rework["blocked_proposal_ids"]

    report = rework["claim_verification_reports"][0]

    assert report["fully_supported"] is False
    assert report["unsupported_claims"]

    corrected_response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "edit",
            "approved_proposal_ids": [],
            "rejected_proposal_ids": [],
            "edits": [
                {
                    "proposal_id": proposal_id,
                    "edited_text": ("Built a Python application using FastAPI."),
                }
            ],
            "reviewer_comment": ("Removed unsupported metric."),
        },
    )

    assert corrected_response.status_code == 200

    completed = corrected_response.json()

    assert completed["status"] == "completed"
    assert completed["review_status"] == "edited"

    assert (
        completed["final_cv_proposals"][0]["proposed_text"]
        == "Built a Python application using FastAPI."
    )


def test_regeneration_returns_to_human_review_before_approval(
    client: TestClient,
    audit_repository: (InMemoryJobAnalysisAuditRepository),
) -> None:
    """A regenerated proposal must receive another human decision."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-REGENERATE",
    )

    thread_id = paused["thread_id"]
    proposal_id = paused["cv_proposals"][0]["proposal_id"]

    regenerate_response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "regenerate",
            "approved_proposal_ids": [],
            "rejected_proposal_ids": [proposal_id],
            "edits": [],
            "reviewer_comment": ("Make the wording more concise."),
        },
    )

    assert regenerate_response.status_code == 200

    regenerated = regenerate_response.json()

    assert regenerated["status"] == "awaiting_review"
    assert regenerated["thread_id"] == thread_id

    assert regenerated["cv_proposals"][0]["proposal_id"] == proposal_id

    assert (
        regenerated["cv_proposals"][0]["proposed_text"]
        == "Built a concise Python API using FastAPI."
    )

    # Passing verification is still not approval.
    assert regenerated["reviewable_proposal_ids"] == [proposal_id]
    assert regenerated["blocked_proposal_ids"] == []

    first_snapshot = audit_repository.get_run(
        user_id="USER-API-001",
        thread_id=thread_id,
    )

    assert first_snapshot is not None
    assert first_snapshot.status.value == "awaiting_review"

    first_reviews = audit_repository.list_reviews(
        user_id="USER-API-001",
        thread_id=thread_id,
    )

    assert len(first_reviews) == 1

    assert first_reviews[0].sequence_number == 1
    assert first_reviews[0].decision.action.value == "regenerate"
    assert first_reviews[0].result_status.value == "awaiting_review"

    assert regenerated["review"]["allowed_actions"] == [
        "approve",
        "edit",
        "regenerate",
        "reject",
    ]

    approve_response = client.post(
        f"/api/v1/job-analysis/{thread_id}/review",
        headers={
            "X-User-ID": "USER-API-001",
        },
        json={
            "action": "approve",
            "approved_proposal_ids": [proposal_id],
            "rejected_proposal_ids": [],
            "edits": [],
            "reviewer_comment": ("Approved regenerated wording."),
        },
    )

    assert approve_response.status_code == 200

    completed = approve_response.json()

    assert completed["status"] == "completed"
    assert completed["review_status"] == "approved"

    assert (
        completed["final_cv_proposals"][0]["proposed_text"]
        == "Built a concise Python API using FastAPI."
    )

    final_snapshot = audit_repository.get_run(
        user_id="USER-API-001",
        thread_id=thread_id,
    )

    assert final_snapshot is not None
    assert final_snapshot.status.value == "completed"
    assert final_snapshot.review_status is not None
    assert final_snapshot.review_status.value == "approved"

    reviews = audit_repository.list_reviews(
        user_id="USER-API-001",
        thread_id=thread_id,
    )

    assert [review.sequence_number for review in reviews] == [1, 2]

    assert [review.decision.action.value for review in reviews] == [
        "regenerate",
        "approve",
    ]

    assert [review.result_status.value for review in reviews] == [
        "awaiting_review",
        "completed",
    ]


def test_job_analysis_can_be_recovered_while_awaiting_review(
    client: TestClient,
) -> None:
    """Paused analysis should be fully recoverable from durable state."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-RECOVER",
    )

    response = client.get(
        f"/api/v1/job-analysis/{paused['thread_id']}",
        headers={
            "X-User-ID": "USER-API-001",
        },
    )

    assert response.status_code == 200
    assert response.json() == paused


def test_other_user_cannot_recover_job_analysis(
    client: TestClient,
) -> None:
    """A user must not recover another user's durable analysis."""

    paused = start_reviewable_analysis(
        client,
        job_id="JOB-API-RECOVER-CROSS-USER",
    )

    response = client.get(
        f"/api/v1/job-analysis/{paused['thread_id']}",
        headers={
            "X-User-ID": "USER-OTHER",
        },
    )

    assert response.status_code == 404
