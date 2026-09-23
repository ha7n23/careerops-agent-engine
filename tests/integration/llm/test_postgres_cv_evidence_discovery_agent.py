"""Live PostgreSQL + LLM proof for CV evidence discovery."""

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
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceProposal,
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
from careerops_agent_engine.infrastructure.llm.factory import (
    create_evidence_discovery_runner,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Live LLM tests are disabled.",
)
def test_configured_model_discovers_approved_cv_evidence_from_postgres() -> None:
    """The configured model should discover approved PostgreSQL evidence."""

    suffix = uuid4().hex[:8].upper()

    user_id = f"USER-LIVE-H2-{suffix}"
    document_id = f"DOC-H2-{suffix}"
    proposal_id = f"EVP-H2-{suffix}"
    evidence_id = f"EVD-H2-{suffix}"
    review_run_id = f"EVR-H2-{suffix}"
    review_id = f"EVH-H2-{suffix}"

    source_excerpt = "Built a FastAPI application using Python and PostgreSQL."

    document = CareerDocument(
        document_id=document_id,
        original_filename="careerops-h2-live.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=128,
        sha256_hex="b" * 64,
        storage_key=(f"documents/live-h2/{document_id}.pdf"),
        status=CareerDocumentStatus.EXTRACTED,
    )

    proposal = CareerEvidenceProposal(
        proposal_id=proposal_id,
        category=EvidenceCategory.PROJECT,
        title="CareerOps FastAPI Platform",
        source_section=CVSection.PROJECTS,
        source_section_order_index=0,
        technologies=[
            "Python",
            "FastAPI",
            "PostgreSQL",
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
        title="CareerOps FastAPI Platform",
        verification_status=(VerificationStatus.APPROVED),
        technologies=[
            "Python",
            "FastAPI",
            "PostgreSQL",
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

    result = EvidenceReviewResult(
        approved_proposal_ids=[
            proposal_id,
        ],
        approved_evidence=[
            approved_evidence,
        ],
    )

    snapshot = CVEvidenceReviewRunSnapshot(
        review_run_id=review_run_id,
        user_id=user_id,
        document_id=document_id,
        status=(CVEvidenceReviewRunStatus.COMPLETED),
        proposals=[
            proposal,
        ],
        overlap_findings=[],
        document_warnings=[],
        review_result=result,
    )

    decision = EvidenceReviewDecision(
        approved_proposal_ids=[
            proposal_id,
        ],
        reviewer_comment=("7H2 live integration proof."),
    )

    review = CVEvidenceReviewAuditEntry(
        review_id=review_id,
        review_run_id=review_run_id,
        sequence_number=1,
        decision=decision,
        result=result,
    )

    document_repository = get_career_document_repository()
    audit_repository = get_cv_evidence_audit_repository()
    evidence_repository = get_evidence_repository()

    try:
        document_repository.save(
            user_id=user_id,
            document=document,
        )

        audit_repository.save_review_result(
            snapshot=snapshot,
            review=review,
        )

        persisted = evidence_repository.get_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        assert persisted == approved_evidence

        runner = create_evidence_discovery_runner(evidence_repository)

        requirement = JobRequirement(
            requirement_id=f"REQ-H2-{suffix}",
            name="FastAPI API development",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected=("Hands-on experience building APIs with FastAPI."),
            importance_score=5,
            source_text=(
                "Candidates must have hands-on FastAPI API development experience."
            ),
        )

        match = runner.discover(
            requirement,
            user_id=user_id,
        )

        assert match.match_strength in {
            MatchStrength.STRONG,
            MatchStrength.PARTIAL,
        }

        assert evidence_id in (match.direct_evidence_ids)

        assert match.gap is (match.match_strength is MatchStrength.PARTIAL)

        print()
        print("=== CareerOps 7H2 live proof ===")
        print(f"user_id: {user_id}")
        print(f"evidence_id: {evidence_id}")
        print(
            "match_strength:",
            match.match_strength.value,
        )
        print(
            "direct_evidence_ids:",
            match.direct_evidence_ids,
        )
        print(
            "related_evidence_ids:",
            match.related_evidence_ids,
        )
        print("gap:", match.gap)
        print(
            "explanation:",
            match.explanation,
        )
        print("CONFIGURED LLM + POSTGRES EVIDENCE DISCOVERY PASSED")

    finally:
        wait_for_all_tracers()

        with get_database_engine().begin() as connection:
            connection.execute(
                text(
                    """
                    DELETE FROM cv_evidence_review_history
                    WHERE review_run_id = :review_run_id
                    """
                ),
                {
                    "review_run_id": review_run_id,
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
                    "review_run_id": review_run_id,
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
