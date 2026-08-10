"""Build immutable CV versions from trusted final assembly output."""

import json
from collections.abc import Iterable
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    StructuredCVAssemblyError,
)
from careerops_agent_engine.application.services.final_cv_assembly import (
    FinalCVAssemblyExecutionResult,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    CVVersionProvenance,
    LLMProvenanceReference,
)


class CVVersionBuilder:
    """Create one immutable assembled CV version with full provenance."""

    def build(
        self,
        *,
        assembly: FinalCVAssemblyExecutionResult,
        version_number: int,
        parent_version_id: str | None,
        template_id: str,
        template_version: str,
        workflow_version: str,
        llm_references: list[LLMProvenanceReference],
    ) -> CVVersion:
        """Build a content-addressed assembled CV version."""

        review_status = assembly.job_run.review_status

        if review_status is None:
            raise StructuredCVAssemblyError(
                "CV version creation requires an approved "
                "or edited human-review outcome."
            )

        if review_status not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.EDITED,
        }:
            raise StructuredCVAssemblyError(
                "CV version creation requires an approved "
                "or edited human-review outcome."
            )

        applied_changes = list(assembly.tailoring_result.applied_changes)

        requirement_ids = collect_change_requirement_ids(applied_changes)

        evidence_ids = collect_change_evidence_ids(applied_changes)

        proposal_ids = [
            proposal.proposal_id for proposal in (assembly.job_run.final_cv_proposals)
        ]

        provenance = CVVersionProvenance(
            job_id=assembly.job_run.job_id,
            thread_id=assembly.job_run.thread_id,
            source_document_id=(assembly.source_document.document_id),
            requirement_ids=requirement_ids,
            supporting_evidence_ids=evidence_ids,
            final_proposal_ids=proposal_ids,
            review_status=review_status,
            template_id=template_id,
            template_version=template_version,
            workflow_version=workflow_version,
            llm_references=list(llm_references),
        )

        cv_version_id = build_cv_version_id(
            version_number=version_number,
            parent_version_id=parent_version_id,
            structured_cv_payload=(
                assembly.tailoring_result.structured_cv.model_dump(mode="json")
            ),
            applied_change_payload=[
                change.model_dump(mode="json") for change in applied_changes
            ],
            provenance_payload=(provenance.model_dump(mode="json")),
        )

        return CVVersion(
            cv_version_id=cv_version_id,
            version_number=version_number,
            parent_version_id=parent_version_id,
            status=CVVersionStatus.ASSEMBLED,
            structured_cv=(assembly.tailoring_result.structured_cv),
            applied_changes=applied_changes,
            provenance=provenance,
            artifacts=[],
        )


def collect_change_requirement_ids(
    changes: list[AppliedCVChange],
) -> list[str]:
    """Collect requirement provenance in stable first-seen order."""

    return collect_unique_identifiers(
        identifier for change in changes for identifier in change.requirement_ids
    )


def collect_change_evidence_ids(
    changes: list[AppliedCVChange],
) -> list[str]:
    """Collect evidence provenance in stable first-seen order."""

    return collect_unique_identifiers(
        identifier
        for change in changes
        for identifier in change.supporting_evidence_ids
    )


def collect_unique_identifiers(
    identifiers: Iterable[str],
) -> list[str]:
    """Collect unique identifiers while preserving order."""

    result: list[str] = []
    seen: set[str] = set()

    for identifier in identifiers:
        if identifier in seen:
            continue

        seen.add(identifier)
        result.append(identifier)

    return result


def build_cv_version_id(
    *,
    version_number: int,
    parent_version_id: str | None,
    structured_cv_payload: dict[str, object],
    applied_change_payload: list[dict[str, object]],
    provenance_payload: dict[str, object],
) -> str:
    """Build a stable content-addressed CV version identifier."""

    payload = {
        "version_number": version_number,
        "parent_version_id": parent_version_id,
        "structured_cv": structured_cv_payload,
        "applied_changes": applied_change_payload,
        "provenance": provenance_payload,
    }

    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    digest = sha256(canonical_json.encode("utf-8")).hexdigest()[:16].upper()

    return f"CVV-{digest}"
