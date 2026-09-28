"""Deterministic proof of the complete Module 1 evidence contract."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from careerops_agent_engine.agents.nodes.job_analysis import (
    calculate_fit,
    create_discover_evidence_node,
)
from careerops_agent_engine.agents.states.job_analysis import JobAnalysisState
from careerops_agent_engine.application.exceptions import (
    CareerEvidenceUnavailableError,
)
from careerops_agent_engine.application.services.cv_document_extraction import (
    CVDocumentExtractionService,
)
from careerops_agent_engine.application.services.cv_document_ingestion import (
    CVDocumentIngestionService,
)
from careerops_agent_engine.application.services.cv_document_preparation import (
    CVDocumentPreparationService,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.application.services.cv_evidence_duplicates import (
    CVEvidenceDuplicateDetector,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    CVEvidenceProposalService,
)
from careerops_agent_engine.application.services.cv_evidence_review import (
    CVEvidenceReviewService,
)
from careerops_agent_engine.application.services.cv_evidence_workflow import (
    CVEvidenceWorkflowService,
)
from careerops_agent_engine.application.services.evidence_registry import (
    EvidenceRegistryService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
    EvidenceCategory,
    EvidenceDuplicateResolutionAction,
    EvidenceLifecycleStatus,
    EvidenceOverlapScope,
    EvidenceSourceType,
    MatchStrength,
)
from careerops_agent_engine.domain.models.document import ParsedCVDocument
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceCandidate,
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceDuplicateResolution,
    EvidenceReviewDecision,
)
from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceHistoryRecord,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
)
from careerops_agent_engine.infrastructure.documents.cv_section_parser import (
    DeterministicCVSectionParser,
)
from careerops_agent_engine.infrastructure.documents.native_document_extractor import (
    NativeDocumentExtractor,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_career_documents import (  # noqa: E501
    SqlAlchemyCareerDocumentRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_cv_evidence_audit import (  # noqa: E501
    SqlAlchemyCVEvidenceAuditRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)
from careerops_agent_engine.infrastructure.storage.local_document_storage import (
    LocalDocumentStorage,
)
from tests.integration.test_cv_evidence_job_analysis_bridge import (
    RepositoryBackedEvidenceDiscoveryRunner,
    build_python_requirement,
    persist_human_approved_cv_evidence,
)

USER_ID = "USER-BRIDGE-001"
OTHER_USER_ID = "USER-CONTRACT-OTHER"
EXISTING_EVIDENCE_ID = "EVD-BRIDGE-PYTHON"
SOURCE_CLAIM = "Built a FastAPI application using Python."


class DeterministicEvidenceExtractor:
    """Return one grounded proposal candidate without an LLM call."""

    def extract(
        self,
        *,
        document: ParsedCVDocument,
    ) -> list[CareerEvidenceCandidate]:
        """Build a candidate from the parsed project section."""

        assert document.source_type is EvidenceSourceType.MANUAL_ENTRY
        assert len(document.sections) == 1
        assert SOURCE_CLAIM in document.sections[0].text

        return [
            CareerEvidenceCandidate(
                category=EvidenceCategory.PROJECT,
                title="Python API Project",
                source_section_order_index=0,
                source_excerpt=SOURCE_CLAIM,
                technologies=["Python", "FastAPI"],
                capabilities=["API development"],
                claims=[SOURCE_CLAIM],
                warnings=[],
            )
        ]


def test_evidence_contract_from_ingestion_to_job_analysis(
    tmp_path: Path,
) -> None:
    """Prove ingestion, review, lifecycle, querying and job-analysis use."""

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    document_repository = SqlAlchemyCareerDocumentRepository(session_factory)
    audit_repository = SqlAlchemyCVEvidenceAuditRepository(session_factory)
    evidence_repository = SqlAlchemyEvidenceRepository(session_factory)
    registry = EvidenceRegistryService(evidence_repository)

    storage = LocalDocumentStorage(tmp_path / "documents")
    ingestion = CVDocumentIngestionService(
        upload_service=CVDocumentUploadService(
            storage=storage,
            max_upload_bytes=5 * 1024 * 1024,
        ),
        repository=document_repository,
        storage=storage,
    )
    workflow = CVEvidenceWorkflowService(
        document_repository=document_repository,
        audit_repository=audit_repository,
        preparation_service=CVDocumentPreparationService(
            extraction_service=CVDocumentExtractionService(
                storage=storage,
                extractor=NativeDocumentExtractor(),
            ),
            section_parser=DeterministicCVSectionParser(),
        ),
        proposal_service=CVEvidenceProposalService(
            extractor=DeterministicEvidenceExtractor(),
        ),
        duplicate_detector=CVEvidenceDuplicateDetector(
            repository=evidence_repository,
        ),
        review_service=CVEvidenceReviewService(
            repository=evidence_repository,
        ),
    )

    try:
        persist_human_approved_cv_evidence(
            document_repository=document_repository,
            audit_repository=audit_repository,
        )

        document = ingestion.ingest_text(
            user_id=USER_ID,
            title="CareerOps contract proof",
            content=f"Projects\n{SOURCE_CLAIM}",
        )
        assert document.status is CareerDocumentStatus.UPLOADED

        pending = workflow.start_review(
            user_id=USER_ID,
            document_id=document.document_id,
        )
        assert pending.status is CVEvidenceReviewRunStatus.AWAITING_REVIEW
        assert len(pending.proposals) == 1
        assert len(pending.overlap_findings) == 1

        proposal = pending.proposals[0]
        finding = pending.overlap_findings[0]
        assert finding.scope is EvidenceOverlapScope.APPROVED_EVIDENCE
        assert finding.matching_evidence_id == EXISTING_EVIDENCE_ID
        assert finding.matched_claims == [SOURCE_CLAIM]

        decision = EvidenceReviewDecision(
            approved_proposal_ids=[proposal.proposal_id],
            duplicate_resolutions=[
                EvidenceDuplicateResolution(
                    proposal_id=proposal.proposal_id,
                    scope=EvidenceOverlapScope.APPROVED_EVIDENCE,
                    action=(EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING),
                    matching_evidence_id=EXISTING_EVIDENCE_ID,
                )
            ],
            reviewer_comment="Deterministic Module 1 contract proof.",
        )
        completed = workflow.submit_review(
            user_id=USER_ID,
            review_run_id=pending.review_run_id,
            decision=decision,
        )
        assert completed.status is CVEvidenceReviewRunStatus.COMPLETED
        assert completed.review_result is not None
        assert len(completed.review_result.evidence_updates) == 1
        assert completed.review_result.approved_evidence == []

        retry = workflow.submit_review(
            user_id=USER_ID,
            review_run_id=pending.review_run_id,
            decision=decision,
        )
        assert retry == completed
        assert (
            len(
                audit_repository.list_reviews(
                    user_id=USER_ID,
                    review_run_id=pending.review_run_id,
                )
            )
            == 1
        )

        merged = registry.get_approved(
            user_id=USER_ID,
            evidence_id=EXISTING_EVIDENCE_ID,
        )
        assert len(merged.source_references) == 2

        active_page = registry.query_approved(
            user_id=USER_ID,
            query="python fastapi",
            category=EvidenceCategory.PROJECT,
            lifecycle_status=EvidenceLifecycleStatus.ACTIVE,
            offset=0,
            limit=10,
        )
        assert active_page.total == 1
        assert active_page.items == [merged]

        archived = registry.archive_approved(
            user_id=USER_ID,
            evidence_id=EXISTING_EVIDENCE_ID,
        )
        assert archived.lifecycle_status is EvidenceLifecycleStatus.ARCHIVED
        assert (
            evidence_repository.search_approved(
                user_id=USER_ID,
                query="Python",
            )
            == []
        )
        assert_job_fit(
            repository=evidence_repository,
            user_id=USER_ID,
            expected_strength=MatchStrength.NONE,
            expected_fit=0.0,
        )

        archived_page = registry.query_approved(
            user_id=USER_ID,
            query="python",
            category=EvidenceCategory.PROJECT,
            lifecycle_status=EvidenceLifecycleStatus.ARCHIVED,
            offset=0,
            limit=10,
        )
        assert archived_page.total == 1
        assert archived_page.items[0].evidence_id == EXISTING_EVIDENCE_ID

        with pytest.raises(
            CareerEvidenceUnavailableError,
            match="approved evidence record is unavailable",
        ):
            registry.get_approved(
                user_id=OTHER_USER_ID,
                evidence_id=EXISTING_EVIDENCE_ID,
            )

        other_user_page = registry.query_approved(
            user_id=OTHER_USER_ID,
            query="python",
            category=None,
            lifecycle_status=EvidenceLifecycleStatus.ACTIVE,
            offset=0,
            limit=10,
        )
        assert other_user_page.total == 0
        assert_job_fit(
            repository=evidence_repository,
            user_id=OTHER_USER_ID,
            expected_strength=MatchStrength.NONE,
            expected_fit=0.0,
        )

        restored = registry.restore_approved(
            user_id=USER_ID,
            evidence_id=EXISTING_EVIDENCE_ID,
        )
        assert restored.lifecycle_status is EvidenceLifecycleStatus.ACTIVE
        assert_job_fit(
            repository=evidence_repository,
            user_id=USER_ID,
            expected_strength=MatchStrength.STRONG,
            expected_fit=100.0,
        )

        persisted_document = document_repository.get(
            user_id=USER_ID,
            document_id=document.document_id,
        )
        assert persisted_document is not None
        assert persisted_document.status is CareerDocumentStatus.EXTRACTED

        with session_factory() as session:
            actions = [
                record.action
                for record in session.query(CareerEvidenceHistoryRecord)
                .order_by(CareerEvidenceHistoryRecord.created_at)
                .all()
            ]
        assert actions == ["edit", "archive", "restore"]
    finally:
        engine.dispose()


def assert_job_fit(
    *,
    repository: SqlAlchemyEvidenceRepository,
    user_id: str,
    expected_strength: MatchStrength,
    expected_fit: float,
) -> None:
    """Run production discovery and fit nodes against current registry state."""

    requirement = build_python_requirement()
    state: JobAnalysisState = {
        "job_id": "JOB-CONTRACT-001",
        "user_id": user_id,
        "job_description": "Strong Python development experience required.",
        "requirements": [requirement.model_dump(mode="json")],
        "audit_events": [],
    }
    discovery_update = create_discover_evidence_node(
        RepositoryBackedEvidenceDiscoveryRunner(repository)
    )(state)
    match_payloads = discovery_update.get("evidence_matches", [])
    state["evidence_matches"] = match_payloads

    matches = [EvidenceMatch.model_validate(payload) for payload in match_payloads]
    assert len(matches) == 1
    assert matches[0].match_strength is expected_strength
    assert calculate_fit(state).get("fit_score") == expected_fit
