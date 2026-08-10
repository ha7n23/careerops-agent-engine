"""Tests for immutable CV-version construction."""

from careerops_agent_engine.application.services.cv_version_builder import (
    CVVersionBuilder,
    collect_change_evidence_ids,
    collect_change_requirement_ids,
)
from careerops_agent_engine.application.services.final_cv_assembly import (
    FinalCVAssemblyExecutionResult,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVChangeApplicationMode,
    CVSection,
    CVVersionStatus,
    JobAnalysisRunStatus,
)
from careerops_agent_engine.domain.models.audit import (
    JobAnalysisRunSnapshot,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
    StructuredCVTailoringResult,
)
from careerops_agent_engine.domain.models.cv_version import (
    LLMProvenanceReference,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)


def build_change() -> AppliedCVChange:
    """Create one exact applied final proposal."""

    return AppliedCVChange(
        change_id="CHG-001",
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
        source_anchor=("Built CareerOps using Python."),
        original_text=("Built CareerOps using Python."),
        applied_text=("Built CareerOps using Python and FastAPI."),
        anchor_evidence_ids=["EVD-001"],
        requirement_ids=["REQ-001"],
        supporting_evidence_ids=["EVD-001"],
    )


def build_assembly() -> FinalCVAssemblyExecutionResult:
    """Create trusted persisted assembly output."""

    proposal = CVChangeProposal(
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        proposed_text=("Built CareerOps using Python and FastAPI."),
        requirement_ids=["REQ-001"],
        supporting_evidence_ids=["EVD-001"],
        confidence_score=1.0,
    )

    job_run = JobAnalysisRunSnapshot(
        thread_id="THR-001",
        user_id="USER-001",
        job_id="JOB-001",
        status=JobAnalysisRunStatus.COMPLETED,
        review_status=ApprovalStatus.APPROVED,
        final_cv_proposals=[proposal],
    )

    source_document = CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key="documents/user/DOC-001.pdf",
        status=CareerDocumentStatus.EXTRACTED,
    )

    structured_cv = StructuredCV(
        cv_id="CV-001",
        source_document_id="DOC-001",
        preamble_text="Example Candidate",
        sections=[
            StructuredCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                free_text=("CareerOps\nBuilt CareerOps using Python and FastAPI."),
            )
        ],
    )

    return FinalCVAssemblyExecutionResult(
        job_run=job_run,
        source_document=source_document,
        approved_evidence=(CareerEvidence.model_construct(evidence_id="EVD-001"),),
        tailoring_result=(
            StructuredCVTailoringResult(
                structured_cv=structured_cv,
                applied_changes=[build_change()],
            )
        ),
    )


def build_llm_references() -> list[LLMProvenanceReference]:
    """Describe the LLM components contributing to final tailoring."""

    return [
        LLMProvenanceReference(
            component="cv_proposal_generation",
            provider="google",
            model="gemini-3.5-flash",
            prompt_version="cv-proposal-v1",
        ),
        LLMProvenanceReference(
            component="cv_claim_verification",
            provider="google",
            model="gemini-3.5-flash",
            prompt_version=("cv-claim-verification-v1"),
        ),
    ]


def test_builder_creates_assembled_version_with_full_provenance() -> None:
    """Trusted assembly should become one immutable CV version."""

    version = CVVersionBuilder().build(
        assembly=build_assembly(),
        version_number=1,
        parent_version_id=None,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="1.0.0",
        llm_references=build_llm_references(),
    )

    assert version.status is CVVersionStatus.ASSEMBLED

    assert version.version_number == 1
    assert version.parent_version_id is None

    assert version.provenance.job_id == "JOB-001"
    assert version.provenance.thread_id == "THR-001"

    assert version.provenance.requirement_ids == ["REQ-001"]

    assert version.provenance.supporting_evidence_ids == ["EVD-001"]

    assert version.provenance.final_proposal_ids == ["CVP-001"]

    assert len(version.applied_changes) == 1
    assert version.artifacts == []


def test_exact_retry_produces_same_version_identifier() -> None:
    """Content-addressed version creation should be idempotent."""

    builder = CVVersionBuilder()

    first = builder.build(
        assembly=build_assembly(),
        version_number=1,
        parent_version_id=None,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="1.0.0",
        llm_references=build_llm_references(),
    )

    second = builder.build(
        assembly=build_assembly(),
        version_number=1,
        parent_version_id=None,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="1.0.0",
        llm_references=build_llm_references(),
    )

    assert first.cv_version_id == second.cv_version_id


def test_meaningful_provenance_change_changes_version_identifier() -> None:
    """Workflow provenance must participate in version identity."""

    builder = CVVersionBuilder()

    first = builder.build(
        assembly=build_assembly(),
        version_number=1,
        parent_version_id=None,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="1.0.0",
        llm_references=build_llm_references(),
    )

    second = builder.build(
        assembly=build_assembly(),
        version_number=1,
        parent_version_id=None,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="2.0.0",
        llm_references=build_llm_references(),
    )

    assert first.cv_version_id != second.cv_version_id


def test_change_identifier_collection_is_stable() -> None:
    """Repeated provenance IDs should appear once in first-seen order."""

    first = build_change()

    second = build_change().model_copy(
        update={
            "change_id": "CHG-002",
            "proposal_id": "CVP-002",
            "requirement_ids": [
                "REQ-002",
                "REQ-001",
            ],
            "supporting_evidence_ids": [
                "EVD-002",
                "EVD-001",
            ],
            "anchor_evidence_ids": ["EVD-002"],
        }
    )

    assert collect_change_requirement_ids(
        [
            first,
            second,
        ]
    ) == [
        "REQ-001",
        "REQ-002",
    ]

    assert collect_change_evidence_ids(
        [
            first,
            second,
        ]
    ) == [
        "EVD-001",
        "EVD-002",
    ]
