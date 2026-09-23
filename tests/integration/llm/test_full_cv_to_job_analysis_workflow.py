"""Live proof that approved CV evidence reaches
the full job-analysis safety boundary."""

import os
from uuid import uuid4

import pytest
from langchain_core.tracers.langchain import wait_for_all_tracers
from sqlalchemy import text

from careerops_agent_engine.api.dependencies import (
    get_career_document_repository,
    get_cv_evidence_audit_repository,
    get_database_engine,
    get_evidence_repository,
    get_job_analysis_audit_repository,
    get_job_analysis_service,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    JobAnalysisRunStatus,
    MatchStrength,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceProposal,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
    EvidenceReviewResult,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
)
from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Live LLM tests are disabled.",
)
def test_full_job_analysis_uses_approved_cv_evidence_safely() -> None:
    """Run real configured-provider job analysis through the safety boundary."""

    suffix = uuid4().hex[:8].upper()

    user_id = f"USER-LIVE-H3-{suffix}"
    document_id = f"DOC-H3-{suffix}"
    proposal_id = f"EVP-H3-{suffix}"
    evidence_id = f"EVD-H3-{suffix}"
    evidence_review_run_id = f"EVR-H3-{suffix}"
    evidence_review_id = f"EVH-H3-{suffix}"
    job_id = f"JOB-LIVE-H3-{suffix}"

    thread_id: str | None = None

    source_excerpt = "Built a production-style Python API using FastAPI."

    document = CareerDocument(
        document_id=document_id,
        original_filename="careerops-h3-live.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=128,
        sha256_hex="c" * 64,
        storage_key=(f"documents/live-h3/{document_id}.pdf"),
        status=CareerDocumentStatus.EXTRACTED,
    )

    evidence_proposal = CareerEvidenceProposal(
        proposal_id=proposal_id,
        category=EvidenceCategory.PROJECT,
        title="Production Python API",
        source_section=CVSection.PROJECTS,
        source_section_order_index=0,
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=[
            "API development",
        ],
        claims=[
            source_excerpt,
        ],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id=document_id,
                source_excerpt=source_excerpt,
            )
        ],
        warnings=[],
    )

    approved_evidence = CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title="Production Python API",
        verification_status=(VerificationStatus.APPROVED),
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=[
            "API development",
        ],
        approved_claims=[
            source_excerpt,
        ],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id=document_id,
                source_excerpt=source_excerpt,
            )
        ],
    )

    evidence_review_result = EvidenceReviewResult(
        approved_proposal_ids=[
            proposal_id,
        ],
        approved_evidence=[
            approved_evidence,
        ],
    )

    evidence_review_snapshot = CVEvidenceReviewRunSnapshot(
        review_run_id=evidence_review_run_id,
        user_id=user_id,
        document_id=document_id,
        status=(CVEvidenceReviewRunStatus.COMPLETED),
        proposals=[
            evidence_proposal,
        ],
        overlap_findings=[],
        document_warnings=[],
        review_result=evidence_review_result,
    )

    evidence_review_decision = EvidenceReviewDecision(
        approved_proposal_ids=[
            proposal_id,
        ],
        reviewer_comment=("7H3 live full-workflow integration proof."),
    )

    evidence_review_audit = CVEvidenceReviewAuditEntry(
        review_id=evidence_review_id,
        review_run_id=evidence_review_run_id,
        sequence_number=1,
        decision=evidence_review_decision,
        result=evidence_review_result,
    )

    job_description = (
        "Junior Python API Engineer. "
        "Essential requirement: hands-on experience building "
        "Python APIs with FastAPI. "
        "This role-specific technical requirement should be "
        "demonstrated through practical project or employment evidence."
    )

    document_repository = get_career_document_repository()
    cv_audit_repository = get_cv_evidence_audit_repository()
    evidence_repository = get_evidence_repository()
    job_audit_repository = get_job_analysis_audit_repository()

    try:
        # ---------------------------------------------------------
        # 1. Persist evidence that has crossed explicit human review.
        # ---------------------------------------------------------
        document_repository.save(
            user_id=user_id,
            document=document,
        )

        cv_audit_repository.save_review_result(
            snapshot=evidence_review_snapshot,
            review=evidence_review_audit,
        )

        persisted_evidence = evidence_repository.get_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        assert persisted_evidence == approved_evidence

        # ---------------------------------------------------------
        # 2. Run the real production job-analysis service.
        #
        # This uses:
        # - real requirement extraction
        # - real agentic evidence discovery
        # - deterministic fit scoring
        # - real CV proposal generation
        # - real claim verification
        # - PostgreSQL LangGraph checkpointing
        # ---------------------------------------------------------
        execution = get_job_analysis_service().analyse(
            job_id=job_id,
            user_id=user_id,
            job_description=job_description,
        )

        thread_id = execution.thread_id

        state = execution.state

        # ---------------------------------------------------------
        # 3. Requirement extraction really ran.
        # ---------------------------------------------------------
        requirement_payloads = state.get(
            "requirements",
            [],
        )

        assert requirement_payloads

        requirements = [
            JobRequirement.model_validate(payload) for payload in requirement_payloads
        ]

        # ---------------------------------------------------------
        # 4. The real evidence agent discovered the CV evidence.
        # ---------------------------------------------------------
        match_payloads = state.get(
            "evidence_matches",
            [],
        )

        assert match_payloads

        matches = [EvidenceMatch.model_validate(payload) for payload in match_payloads]

        direct_matches = [
            match for match in matches if evidence_id in match.direct_evidence_ids
        ]

        assert direct_matches

        assert any(
            match.match_strength
            in {
                MatchStrength.STRONG,
                MatchStrength.PARTIAL,
            }
            for match in direct_matches
        )

        # ---------------------------------------------------------
        # 5. Deterministic fit scoring used those evidence matches.
        # ---------------------------------------------------------
        fit_score = state.get("fit_score")

        assert fit_score is not None
        assert fit_score > 0.0
        assert fit_score <= 100.0

        # ---------------------------------------------------------
        # 6. Real proposal generation cited approved CV evidence.
        # ---------------------------------------------------------
        proposal_payloads = state.get(
            "cv_proposals",
            [],
        )

        assert proposal_payloads

        cv_proposals = [
            CVChangeProposal.model_validate(payload) for payload in proposal_payloads
        ]

        evidence_backed_proposals = [
            proposal
            for proposal in cv_proposals
            if evidence_id in proposal.supporting_evidence_ids
        ]

        assert evidence_backed_proposals

        # ---------------------------------------------------------
        # 7. Real claim verification produced a reviewable proposal.
        # ---------------------------------------------------------
        verification_payloads = state.get(
            "claim_verification_reports",
            [],
        )

        assert verification_payloads

        reports = [
            CVClaimVerificationReport.model_validate(payload)
            for payload in verification_payloads
        ]

        fully_supported_ids = {
            report.proposal_id for report in reports if report.fully_supported
        }

        unsupported_report_ids = {
            report.proposal_id
            for report in reports
            if (not report.fully_supported and report.unsupported_claims)
        }

        reviewable_ids = set(
            state.get(
                "reviewable_proposal_ids",
                [],
            )
        )

        blocked_ids = set(
            state.get(
                "blocked_proposal_ids",
                [],
            )
        )

        evidence_backed_ids = {
            proposal.proposal_id for proposal in evidence_backed_proposals
        }

        assert reviewable_ids <= fully_supported_ids
        assert blocked_ids <= unsupported_report_ids

        # Every proposal generated from the newly approved CV evidence
        # must end at one of the two safe boundaries:
        #
        # 1. fully supported -> human review interrupt
        # 2. unsupported -> deterministic blocking
        assert evidence_backed_ids & (reviewable_ids | blocked_ids)

        # ---------------------------------------------------------
        # 8. Accept either valid safety outcome from the live LLM.
        # ---------------------------------------------------------
        if execution.awaiting_review:
            interrupt_payload = execution.interrupt_payload

            assert interrupt_payload is not None
            assert reviewable_ids
            assert evidence_backed_ids & reviewable_ids

            assert interrupt_payload["type"] == ("cv_proposal_review")

            interrupt_proposals = interrupt_payload.get("proposals")

            assert isinstance(
                interrupt_proposals,
                list,
            )
            assert interrupt_proposals

            expected_run_status = JobAnalysisRunStatus.AWAITING_REVIEW

            safety_outcome = "fully supported proposal reached human-review interrupt"

        else:
            assert execution.interrupt_payload is None
            assert state.get("status") == "completed"
            assert blocked_ids
            assert evidence_backed_ids & blocked_ids

            expected_run_status = JobAnalysisRunStatus.COMPLETED

            safety_outcome = "unsupported proposal was deterministically blocked"

        # ---------------------------------------------------------
        # 9. Business snapshot was also durably persisted.
        # ---------------------------------------------------------
        persisted_job_run = job_audit_repository.get_run(
            user_id=user_id,
            thread_id=thread_id,
        )

        assert persisted_job_run is not None

        assert persisted_job_run.status is expected_run_status

        assert persisted_job_run.fit_score == fit_score

        assert set(persisted_job_run.reviewable_proposal_ids) == reviewable_ids

        assert set(persisted_job_run.blocked_proposal_ids) == blocked_ids

        print()
        print("=== CareerOps 7H3 full live workflow proof ===")
        print(f"user_id: {user_id}")
        print(f"job_id: {job_id}")
        print(f"thread_id: {thread_id}")
        print(
            "role_title:",
            state.get("role_title"),
        )
        print(
            "requirements:",
            len(requirements),
        )

        for requirement in requirements:
            print(
                "  requirement:",
                requirement.requirement_id,
                "|",
                requirement.name,
                "|",
                requirement.category.value,
            )

        print(
            "evidence matches:",
            len(matches),
        )

        for match in matches:
            print(
                "  match:",
                match.requirement_id,
                "|",
                match.match_strength.value,
                "| direct:",
                match.direct_evidence_ids,
                "| gap:",
                match.gap,
            )

        print("fit_score:", fit_score)

        print(
            "CV proposals:",
            len(cv_proposals),
        )

        for proposal in cv_proposals:
            print(
                "  proposal:",
                proposal.proposal_id,
                "| evidence:",
                proposal.supporting_evidence_ids,
            )

        print(
            "reviewable proposal IDs:",
            sorted(reviewable_ids),
        )
        print(
            "blocked proposal IDs:",
            sorted(blocked_ids),
        )
        print(
            "safety outcome:",
            safety_outcome,
        )

        if execution.interrupt_payload is not None:
            print(
                "interrupt type:",
                execution.interrupt_payload["type"],
            )

        print("FULL APPROVED-CV-EVIDENCE JOB-ANALYSIS SAFETY WORKFLOW PASSED")

    finally:
        wait_for_all_tracers()

        # ---------------------------------------------------------
        # Clean LangGraph execution state first.
        # ---------------------------------------------------------
        if thread_id is not None:
            with open_postgres_checkpointer() as checkpointer:
                checkpointer.delete_thread(thread_id)

        # ---------------------------------------------------------
        # Clean only rows created by this unique live test.
        # ---------------------------------------------------------
        with get_database_engine().begin() as connection:
            if thread_id is not None:
                connection.execute(
                    text(
                        """
                        DELETE FROM cv_review_history
                        WHERE thread_id = :thread_id
                        """
                    ),
                    {
                        "thread_id": thread_id,
                    },
                )

                connection.execute(
                    text(
                        """
                        DELETE FROM job_analysis_runs
                        WHERE thread_id = :thread_id
                        """
                    ),
                    {
                        "thread_id": thread_id,
                    },
                )

            connection.execute(
                text(
                    """
                    DELETE FROM cv_evidence_review_history
                    WHERE review_run_id = :review_run_id
                    """
                ),
                {
                    "review_run_id": (evidence_review_run_id),
                },
            )

            connection.execute(
                text(
                    """
                    DELETE FROM career_evidence
                    WHERE user_id = :user_id
                      AND evidence_id = :evidence_id
                    """
                ),
                {
                    "user_id": user_id,
                    "evidence_id": evidence_id,
                },
            )

            connection.execute(
                text(
                    """
                    DELETE FROM cv_evidence_review_runs
                    WHERE review_run_id = :review_run_id
                    """
                ),
                {
                    "review_run_id": (evidence_review_run_id),
                },
            )

            connection.execute(
                text(
                    """
                    DELETE FROM career_documents
                    WHERE user_id = :user_id
                      AND document_id = :document_id
                    """
                ),
                {
                    "user_id": user_id,
                    "document_id": document_id,
                },
            )
