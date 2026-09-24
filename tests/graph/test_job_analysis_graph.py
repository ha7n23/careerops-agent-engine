"""Tests for the durable evidence-grounded job-analysis graph."""

from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from careerops_agent_engine.agents.graphs.job_analysis import (
    build_job_analysis_graph,
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
    ProposalEdit,
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
        """Regenerate wording according to controlled feedback."""

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
    """Verify safe wording and reject an invented latency metric."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return a report based on the proposal's current text."""

        assert approved_evidence

        lowered_text = proposal.proposed_text.casefold()

        has_invented_metric = "70" in lowered_text and "latency" in lowered_text

        if has_invented_metric:
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
                            "Approved evidence contains "
                            "Python and FastAPI experience, "
                            "but no Kubernetes experience."
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
                    explanation=(
                        "The approved project evidence directly supports this wording."
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
        "edit",
        "regenerate",
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


def test_human_edit_is_reverified_before_completion() -> None:
    """A grounded human edit should be verified before finalization."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "THREAD-GRAPH-SAFE-EDIT",
        }
    }

    paused = graph.invoke(
        {
            "job_id": "JOB-SAFE-EDIT",
            "user_id": "USER-TEST-001",
            "job_description": (
                "Strong Python development experience is required for this role."
            ),
            "audit_events": [],
        },
        config=config,
    )

    proposal = CVChangeProposal.model_validate(paused["cv_proposals"][0])

    edited_text = "Built a Python application using FastAPI."

    decision = CVReviewDecision(
        action=ReviewAction.EDIT,
        edits=[
            ProposalEdit(
                proposal_id=proposal.proposal_id,
                edited_text=edited_text,
            )
        ],
    )

    resumed = graph.invoke(
        Command(resume=decision.model_dump(mode="json")),
        config=config,
    )

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "completed"
    assert resumed["review_status"] == "edited"

    assert resumed["edit_verification_failed_ids"] == []
    assert resumed["reviewable_proposal_ids"] == [proposal.proposal_id]

    assert len(resumed["final_cv_proposals"]) == 1

    final_proposal = CVChangeProposal.model_validate(resumed["final_cv_proposals"][0])

    assert final_proposal.proposed_text == edited_text

    assert "human_edits_applied" in [
        event["event"] for event in resumed["audit_events"]
    ]

    assert "human_edits_reverified" in [
        event["event"] for event in resumed["audit_events"]
    ]


def test_unsupported_human_edit_requires_rework() -> None:
    """An invented human metric must trigger another interrupt."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "THREAD-GRAPH-UNSAFE-EDIT",
        }
    }

    paused = graph.invoke(
        {
            "job_id": "JOB-UNSAFE-EDIT",
            "user_id": "USER-TEST-001",
            "job_description": (
                "Strong Python development experience is required for this role."
            ),
            "audit_events": [],
        },
        config=config,
    )

    proposal = CVChangeProposal.model_validate(paused["cv_proposals"][0])

    unsafe_decision = CVReviewDecision(
        action=ReviewAction.EDIT,
        edits=[
            ProposalEdit(
                proposal_id=proposal.proposal_id,
                edited_text=(
                    "Built a Python FastAPI application and "
                    "reduced production latency by 70 percent."
                ),
            )
        ],
    )

    rework = graph.invoke(
        Command(resume=unsafe_decision.model_dump(mode="json")),
        config=config,
    )

    assert "__interrupt__" in rework
    assert rework["edit_verification_failed_ids"] == [proposal.proposal_id]
    assert rework["reviewable_proposal_ids"] == []
    assert proposal.proposal_id in (rework["blocked_proposal_ids"])

    interrupts = rework["__interrupt__"]

    assert len(interrupts) == 1

    payload = interrupts[0].value

    assert payload["type"] == "cv_proposal_review"
    assert payload["allowed_actions"] == [
        "edit",
        "regenerate",
        "reject",
    ]

    verification_report = CVClaimVerificationReport.model_validate(
        payload["verification_reports"][0]
    )

    assert verification_report.fully_supported is False
    assert verification_report.unsupported_claims == [
        "Reduced production latency by 70 percent."
    ]

    corrected_decision = CVReviewDecision(
        action=ReviewAction.EDIT,
        edits=[
            ProposalEdit(
                proposal_id=proposal.proposal_id,
                edited_text=("Built a Python application using FastAPI."),
            )
        ],
    )

    completed = graph.invoke(
        Command(resume=corrected_decision.model_dump(mode="json")),
        config=config,
    )

    assert completed["status"] == "completed"
    assert completed["review_status"] == "edited"
    assert completed["edit_verification_failed_ids"] == []

    final_proposal = CVChangeProposal.model_validate(completed["final_cv_proposals"][0])

    assert final_proposal.proposed_text == ("Built a Python application using FastAPI.")

    events = [event["event"] for event in completed["audit_events"]]

    # Verification occurred once after each human edit.
    assert events.count("human_edits_reverified") == 2

    assert "human_edit_review_received" in events


def test_supported_regeneration_requires_human_review_again() -> None:
    """A safe regeneration must never auto-approve itself."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": ("THREAD-GRAPH-REGENERATE-SAFE"),
        }
    }

    paused = graph.invoke(
        {
            "job_id": "JOB-REGENERATE-SAFE",
            "user_id": "USER-TEST-001",
            "job_description": (
                "Strong Python development experience is required for this role."
            ),
            "audit_events": [],
        },
        config=config,
    )

    proposal = CVChangeProposal.model_validate(paused["cv_proposals"][0])

    regenerate_decision = CVReviewDecision(
        action=ReviewAction.REGENERATE,
        rejected_proposal_ids=[proposal.proposal_id],
        reviewer_comment=("Make the wording more concise."),
    )

    regenerated = graph.invoke(
        Command(resume=regenerate_decision.model_dump(mode="json")),
        config=config,
    )

    # Regeneration passed verification but MUST pause again.
    assert "__interrupt__" in regenerated
    assert regenerated.get("status") != "completed"

    assert regenerated["regenerated_proposal_ids"] == [proposal.proposal_id]
    assert regenerated["regeneration_verification_failed_ids"] == []

    assert regenerated["reviewable_proposal_ids"] == [proposal.proposal_id]
    assert regenerated["blocked_proposal_ids"] == []

    regenerated_proposal = CVChangeProposal.model_validate(
        regenerated["cv_proposals"][0]
    )

    assert regenerated_proposal.proposal_id == proposal.proposal_id
    assert regenerated_proposal.proposed_text == (
        "Built a concise Python API using FastAPI."
    )

    payload = regenerated["__interrupt__"][0].value

    assert payload["allowed_actions"] == [
        "approve",
        "edit",
        "regenerate",
        "reject",
    ]

    events = [event["event"] for event in regenerated["audit_events"]]

    assert "cv_proposals_regenerated" in events
    assert "regenerated_proposals_verified" in events
    assert "human_review_finalized" not in events

    # Human approval is still required after regeneration.
    approve_decision = CVReviewDecision(
        action=ReviewAction.APPROVE,
        approved_proposal_ids=[proposal.proposal_id],
    )

    completed = graph.invoke(
        Command(resume=approve_decision.model_dump(mode="json")),
        config=config,
    )

    assert completed["status"] == "completed"
    assert completed["review_status"] == "approved"

    final_proposal = CVChangeProposal.model_validate(completed["final_cv_proposals"][0])

    assert final_proposal.proposed_text == ("Built a concise Python API using FastAPI.")


def test_unsupported_regeneration_feedback_is_not_evidence() -> None:
    """Reviewer feedback cannot manufacture new experience."""

    checkpointer = InMemorySaver()
    graph = build_test_graph(checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": ("THREAD-GRAPH-REGENERATE-UNSAFE"),
        }
    }

    paused = graph.invoke(
        {
            "job_id": "JOB-REGENERATE-UNSAFE",
            "user_id": "USER-TEST-001",
            "job_description": (
                "Strong Python development experience is required for this role."
            ),
            "audit_events": [],
        },
        config=config,
    )

    proposal = CVChangeProposal.model_validate(paused["cv_proposals"][0])

    decision = CVReviewDecision(
        action=ReviewAction.REGENERATE,
        rejected_proposal_ids=[proposal.proposal_id],
        reviewer_comment=("Add Kubernetes experience and make it stronger."),
    )

    rework = graph.invoke(
        Command(resume=decision.model_dump(mode="json")),
        config=config,
    )

    assert "__interrupt__" in rework

    assert rework["regeneration_verification_failed_ids"] == [proposal.proposal_id]

    assert rework["reviewable_proposal_ids"] == []
    assert proposal.proposal_id in (rework["blocked_proposal_ids"])

    report = CVClaimVerificationReport.model_validate(
        rework["claim_verification_reports"][0]
    )

    assert report.fully_supported is False
    assert "Kubernetes" in (report.unsupported_claims[0])

    payload = rework["__interrupt__"][0].value

    assert payload["allowed_actions"] == [
        "edit",
        "regenerate",
        "reject",
    ]
