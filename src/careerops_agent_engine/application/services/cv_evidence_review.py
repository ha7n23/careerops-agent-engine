"""Human approval boundary for extracted CV evidence."""

from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVEvidenceProposalValidationError,
    CVEvidenceReviewValidationError,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    validate_claim_grounding,
    validate_string_list,
    validate_technology_grounding,
)
from careerops_agent_engine.domain.enums import (
    EvidenceDuplicateResolutionAction,
    EvidenceOverlapScope,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceCandidate,
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
    SourceReference,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceDuplicateResolution,
    EvidenceDuplicateUpdate,
    EvidenceProposalEdit,
    EvidenceReviewDecision,
    EvidenceReviewResult,
)


class CVEvidenceReviewService:
    """Convert explicitly accepted proposals into approved evidence."""

    def __init__(self, repository: EvidenceRepository | None = None) -> None:
        """Store the evidence reader used to validate user-owned targets."""

        self._repository = repository

    def review(
        self,
        *,
        user_id: str | None = None,
        proposals: list[CareerEvidenceProposal],
        overlap_findings: list[CareerEvidenceOverlapFinding],
        decision: EvidenceReviewDecision,
    ) -> EvidenceReviewResult:
        """Validate one complete human decision and apply it."""

        proposals_by_id = build_proposal_index(proposals)

        validate_review_coverage(
            proposals_by_id=proposals_by_id,
            decision=decision,
        )

        resolutions_by_proposal = validate_duplicate_resolutions(
            proposals_by_id=proposals_by_id,
            overlap_findings=overlap_findings,
            decision=decision,
        )

        approved_evidence: list[CareerEvidence] = []
        evidence_updates: list[EvidenceDuplicateUpdate] = []

        edits_by_id = {edit.proposal_id: edit for edit in decision.edits}

        for proposal_id in [
            *decision.approved_proposal_ids,
            *edits_by_id,
        ]:
            proposal = proposals_by_id[proposal_id]
            accepted = approve_proposal(
                proposal=proposal,
                edit=edits_by_id.get(proposal_id),
            )
            resolutions = resolutions_by_proposal.get(proposal_id, [])
            mutation = next(
                (
                    resolution
                    for resolution in resolutions
                    if resolution.action
                    in {
                        EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
                        EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
                    }
                ),
                None,
            )

            if mutation is None:
                approved_evidence.append(accepted)
                continue

            existing = self._get_resolution_target(
                user_id=user_id,
                resolution=mutation,
            )

            updated = (
                replace_existing_evidence(existing=existing, accepted=accepted)
                if mutation.action is EvidenceDuplicateResolutionAction.REPLACE_EXISTING
                else merge_evidence(existing=existing, accepted=accepted)
            )
            evidence_updates.append(
                EvidenceDuplicateUpdate(
                    proposal_id=proposal_id,
                    action=mutation.action,
                    before=existing,
                    after=updated,
                )
            )

        validate_approved_targets(
            repository=self._repository,
            user_id=user_id,
            overlap_findings=overlap_findings,
        )

        return EvidenceReviewResult(
            approved_proposal_ids=list(decision.approved_proposal_ids),
            edited_proposal_ids=[edit.proposal_id for edit in decision.edits],
            rejected_proposal_ids=list(decision.rejected_proposal_ids),
            duplicate_resolutions=list(decision.duplicate_resolutions),
            approved_evidence=approved_evidence,
            evidence_updates=evidence_updates,
        )

    def _get_resolution_target(
        self,
        *,
        user_id: str | None,
        resolution: EvidenceDuplicateResolution,
    ) -> CareerEvidence:
        """Load one active approved mutation target inside its owner boundary."""

        if (
            self._repository is None
            or user_id is None
            or resolution.matching_evidence_id is None
        ):
            raise CVEvidenceReviewValidationError(
                "Duplicate resolution requires a user-scoped evidence repository."
            )

        existing = self._repository.get_approved(
            user_id=user_id,
            evidence_id=resolution.matching_evidence_id,
        )

        if existing is None:
            raise CVEvidenceReviewValidationError(
                "Duplicate resolution target is unavailable for this user."
            )

        return existing


def build_proposal_index(
    proposals: list[CareerEvidenceProposal],
) -> dict[str, CareerEvidenceProposal]:
    """Build a unique review lookup."""

    indexed = {proposal.proposal_id: proposal for proposal in proposals}

    if len(indexed) != len(proposals):
        raise CVEvidenceReviewValidationError(
            "Evidence review contains duplicate proposal identifiers."
        )

    if not indexed:
        raise CVEvidenceReviewValidationError(
            "Evidence review requires at least one pending proposal."
        )

    return indexed


def validate_review_coverage(
    *,
    proposals_by_id: dict[
        str,
        CareerEvidenceProposal,
    ],
    decision: EvidenceReviewDecision,
) -> None:
    """Require every proposal to receive exactly one decision."""

    expected_ids = set(proposals_by_id)

    approved_ids = set(decision.approved_proposal_ids)

    rejected_ids = set(decision.rejected_proposal_ids)

    edited_ids = {edit.proposal_id for edit in decision.edits}

    referenced_ids = approved_ids | rejected_ids | edited_ids

    unknown_ids = referenced_ids - expected_ids

    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))

        raise CVEvidenceReviewValidationError(
            "Evidence review references proposals "
            "outside the current review set: "
            f"{unknown_display}"
        )

    missing_ids = expected_ids - referenced_ids

    if missing_ids:
        missing_display = ", ".join(sorted(missing_ids))

        raise CVEvidenceReviewValidationError(
            "Every evidence proposal must receive "
            "a human decision. Missing: "
            f"{missing_display}"
        )


def validate_duplicate_resolutions(
    *,
    proposals_by_id: dict[
        str,
        CareerEvidenceProposal,
    ],
    overlap_findings: list[CareerEvidenceOverlapFinding],
    decision: EvidenceReviewDecision,
) -> dict[str, list[EvidenceDuplicateResolution]]:
    """Require one valid, outcome-consistent action for every overlap."""

    expected_ids = set(proposals_by_id)

    finding_proposal_ids = {finding.proposal_id for finding in overlap_findings}

    unknown_findings = finding_proposal_ids - expected_ids

    if unknown_findings:
        raise CVEvidenceReviewValidationError(
            "Evidence overlap findings reference "
            "proposals outside the current review set."
        )

    matching_proposal_ids = {
        finding.matching_proposal_id
        for finding in overlap_findings
        if finding.scope is EvidenceOverlapScope.WITHIN_DOCUMENT
    }

    if matching_proposal_ids - expected_ids:
        raise CVEvidenceReviewValidationError(
            "Evidence overlap findings reference a matching proposal outside "
            "the current review set."
        )

    accepted_ids = {
        *decision.approved_proposal_ids,
        *(edit.proposal_id for edit in decision.edits),
    }

    finding_keys = {overlap_key(finding) for finding in overlap_findings}
    resolution_keys = {
        overlap_key(resolution) for resolution in decision.duplicate_resolutions
    }

    if resolution_keys - finding_keys:
        raise CVEvidenceReviewValidationError(
            "Duplicate resolution references an overlap outside the current review."
        )

    if len(finding_keys) != len(overlap_findings):
        raise CVEvidenceReviewValidationError(
            "Evidence review contains duplicate overlap findings."
        )

    if finding_keys - resolution_keys:
        raise CVEvidenceReviewValidationError(
            "Every evidence overlap requires an explicit duplicate resolution."
        )

    resolutions_by_proposal: dict[str, list[EvidenceDuplicateResolution]] = {}

    for resolution in decision.duplicate_resolutions:
        resolutions_by_proposal.setdefault(
            resolution.proposal_id,
            [],
        ).append(resolution)

    for proposal_id in finding_proposal_ids:
        resolutions = resolutions_by_proposal[proposal_id]
        actions = [resolution.action for resolution in resolutions]
        is_accepted = proposal_id in accepted_ids

        if not is_accepted:
            if any(
                action is not EvidenceDuplicateResolutionAction.KEEP_EXISTING
                for action in actions
            ):
                raise CVEvidenceReviewValidationError(
                    "A rejected duplicate proposal must keep the existing evidence."
                )

            for resolution in resolutions:
                if (
                    resolution.scope is EvidenceOverlapScope.WITHIN_DOCUMENT
                    and resolution.matching_proposal_id not in accepted_ids
                ):
                    raise CVEvidenceReviewValidationError(
                        "Keeping a pending duplicate requires accepting its "
                        "matching proposal."
                    )

            continue

        if any(
            resolution.scope is EvidenceOverlapScope.WITHIN_DOCUMENT
            and resolution.action is EvidenceDuplicateResolutionAction.KEEP_EXISTING
            for resolution in resolutions
        ):
            raise CVEvidenceReviewValidationError(
                "An accepted proposal cannot defer to a matching pending proposal."
            )

        mutations = [
            action
            for action in actions
            if action
            in {
                EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
                EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
            }
        ]
        accepts_separate = EvidenceDuplicateResolutionAction.ACCEPT_SEPARATE in actions

        if accepts_separate and mutations:
            raise CVEvidenceReviewValidationError(
                "A duplicate proposal cannot be accepted separately and mutate "
                "existing evidence."
            )

        if len(mutations) > 1:
            raise CVEvidenceReviewValidationError(
                "A duplicate proposal may update only one existing evidence record."
            )

        if not accepts_separate and not mutations:
            raise CVEvidenceReviewValidationError(
                "An accepted overlap requires accept-separate, replace, or merge."
            )

    mutation_target_ids = [
        resolution.matching_evidence_id
        for resolution in decision.duplicate_resolutions
        if resolution.action
        in {
            EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
            EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
        }
    ]

    if len(mutation_target_ids) != len(set(mutation_target_ids)):
        raise CVEvidenceReviewValidationError(
            "One review may update each existing evidence record only once."
        )

    return resolutions_by_proposal


def overlap_key(
    finding: CareerEvidenceOverlapFinding | EvidenceDuplicateResolution,
) -> tuple[str, EvidenceOverlapScope, str | None, str | None]:
    """Build the stable identity shared by a finding and its resolution."""

    return (
        finding.proposal_id,
        finding.scope,
        finding.matching_proposal_id,
        finding.matching_evidence_id,
    )


def validate_approved_targets(
    *,
    repository: EvidenceRepository | None,
    user_id: str | None,
    overlap_findings: list[CareerEvidenceOverlapFinding],
) -> None:
    """Ensure every approved-evidence target is active and user-owned."""

    evidence_ids = {
        finding.matching_evidence_id
        for finding in overlap_findings
        if finding.scope is EvidenceOverlapScope.APPROVED_EVIDENCE
    }

    if not evidence_ids:
        return

    if repository is None or user_id is None:
        raise CVEvidenceReviewValidationError(
            "Duplicate resolution requires a user-scoped evidence repository."
        )

    for evidence_id in evidence_ids:
        if (
            evidence_id is None
            or repository.get_approved(
                user_id=user_id,
                evidence_id=evidence_id,
            )
            is None
        ):
            raise CVEvidenceReviewValidationError(
                "Duplicate resolution target is unavailable for this user."
            )


def approve_proposal(
    *,
    proposal: CareerEvidenceProposal,
    edit: EvidenceProposalEdit | None,
) -> CareerEvidence:
    """Apply optional human edits and cross the approval boundary."""

    title = (
        edit.title if edit is not None and edit.title is not None else proposal.title
    )

    technologies = (
        list(edit.technologies)
        if edit is not None and edit.technologies is not None
        else list(proposal.technologies)
    )

    capabilities = (
        list(edit.capabilities)
        if edit is not None and edit.capabilities is not None
        else list(proposal.capabilities)
    )

    claims = (
        list(edit.claims)
        if edit is not None and edit.claims is not None
        else list(proposal.claims)
    )

    validate_accepted_grounding(
        proposal=proposal,
        title=title,
        technologies=technologies,
        capabilities=capabilities,
        claims=claims,
    )

    return CareerEvidence(
        evidence_id=build_approved_evidence_id(proposal.proposal_id),
        category=proposal.category,
        title=title,
        verification_status=(VerificationStatus.APPROVED),
        technologies=technologies,
        capabilities=capabilities,
        approved_claims=claims,
        source_references=list(proposal.source_references),
    )


def validate_accepted_grounding(
    *,
    proposal: CareerEvidenceProposal,
    title: str,
    technologies: list[str],
    capabilities: list[str],
    claims: list[str],
) -> None:
    """Revalidate accepted factual content before approval."""

    excerpts = [
        reference.source_excerpt
        for reference in proposal.source_references
        if reference.source_excerpt
    ]

    if len(excerpts) != 1:
        raise CVEvidenceReviewValidationError(
            "CV evidence approval requires exactly one source excerpt."
        )

    candidate = CareerEvidenceCandidate(
        category=proposal.category,
        title=title,
        source_section_order_index=(proposal.source_section_order_index),
        source_excerpt=excerpts[0],
        technologies=technologies,
        capabilities=capabilities,
        claims=claims,
        warnings=[],
    )

    try:
        validate_string_list(
            values=candidate.technologies,
            field_name="technologies",
        )

        validate_string_list(
            values=candidate.capabilities,
            field_name="capabilities",
        )

        validate_string_list(
            values=candidate.claims,
            field_name="claims",
        )

        validate_claim_grounding(candidate=candidate)

        validate_technology_grounding(candidate=candidate)

    except CVEvidenceProposalValidationError as exc:
        raise CVEvidenceReviewValidationError(str(exc)) from exc


def build_approved_evidence_id(
    proposal_id: str,
) -> str:
    """Build a stable approved-evidence identifier."""

    digest = sha256(proposal_id.encode("utf-8")).hexdigest()[:16].upper()

    return f"EVD-{digest}"


def replace_existing_evidence(
    *,
    existing: CareerEvidence,
    accepted: CareerEvidence,
) -> CareerEvidence:
    """Replace content while preserving identity and complete provenance."""

    return accepted.model_copy(
        update={
            "evidence_id": existing.evidence_id,
            "lifecycle_status": existing.lifecycle_status,
            "source_references": merge_source_references(
                existing.source_references,
                accepted.source_references,
            ),
        }
    )


def merge_evidence(
    *,
    existing: CareerEvidence,
    accepted: CareerEvidence,
) -> CareerEvidence:
    """Deterministically combine supported fields into existing evidence."""

    return existing.model_copy(
        update={
            "technologies": merge_strings(
                existing.technologies,
                accepted.technologies,
            ),
            "capabilities": merge_strings(
                existing.capabilities,
                accepted.capabilities,
            ),
            "approved_claims": merge_strings(
                existing.approved_claims,
                accepted.approved_claims,
            ),
            "source_references": merge_source_references(
                existing.source_references,
                accepted.source_references,
            ),
        }
    )


def merge_strings(first: list[str], second: list[str]) -> list[str]:
    """Return a stable case-insensitive union without rewriting values."""

    merged: list[str] = []
    seen: set[str] = set()

    for value in [*first, *second]:
        key = value.strip().casefold()

        if key not in seen:
            merged.append(value)
            seen.add(key)

    return merged


def merge_source_references(
    first: list[SourceReference],
    second: list[SourceReference],
) -> list[SourceReference]:
    """Return a stable union of exact provenance records."""

    merged: list[SourceReference] = []
    seen: set[tuple[str, str, int | None, str | None]] = set()

    for reference in [*first, *second]:
        key = (
            reference.source_type.value,
            reference.source_id,
            reference.page_number,
            reference.source_excerpt,
        )

        if key not in seen:
            merged.append(reference)
            seen.add(key)

    return merged
