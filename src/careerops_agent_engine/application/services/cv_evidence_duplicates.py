"""Deterministic duplicate detection for pending CV evidence."""

import re
import unicodedata

from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.domain.enums import (
    EvidenceOverlapScope,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
    SourceReference,
)

APPROVED_EVIDENCE_SCAN_LIMIT = 1_000

NON_WORD_PATTERN = re.compile(
    r"[^\w+#]+",
    flags=re.UNICODE,
)


class CVEvidenceDuplicateDetector:
    """Find high-confidence overlaps without discarding proposals."""

    def __init__(
        self,
        *,
        repository: EvidenceRepository,
    ) -> None:
        """Store the approved-evidence repository."""

        self._repository = repository

    def detect(
        self,
        *,
        user_id: str,
        proposals: list[CareerEvidenceProposal],
    ) -> list[CareerEvidenceOverlapFinding]:
        """Detect overlaps within the CV and approved registry."""

        findings = detect_within_document(proposals)

        approved_evidence = self._repository.list_approved(
            user_id=user_id,
            limit=APPROVED_EVIDENCE_SCAN_LIMIT,
        )

        findings.extend(
            detect_approved_evidence_overlaps(
                proposals=proposals,
                approved_evidence=approved_evidence,
            )
        )

        return findings


def detect_within_document(
    proposals: list[CareerEvidenceProposal],
) -> list[CareerEvidenceOverlapFinding]:
    """Compare each proposal with earlier proposals from the CV."""

    findings: list[CareerEvidenceOverlapFinding] = []

    for index, proposal in enumerate(proposals):
        for earlier_proposal in proposals[:index]:
            matched_claims = find_matching_claims(
                proposal.claims,
                earlier_proposal.claims,
            )

            same_source_excerpt = share_source_excerpt(
                proposal.source_references,
                earlier_proposal.source_references,
            )

            if not matched_claims and not same_source_excerpt:
                continue

            findings.append(
                CareerEvidenceOverlapFinding(
                    proposal_id=proposal.proposal_id,
                    scope=(EvidenceOverlapScope.WITHIN_DOCUMENT),
                    matching_proposal_id=(earlier_proposal.proposal_id),
                    matched_claims=matched_claims,
                    same_source_excerpt=(same_source_excerpt),
                )
            )

    return findings


def detect_approved_evidence_overlaps(
    *,
    proposals: list[CareerEvidenceProposal],
    approved_evidence: list[CareerEvidence],
) -> list[CareerEvidenceOverlapFinding]:
    """Compare pending proposals with existing approved evidence."""

    findings: list[CareerEvidenceOverlapFinding] = []

    for proposal in proposals:
        for evidence in approved_evidence:
            matched_claims = find_matching_claims(
                proposal.claims,
                evidence.approved_claims,
            )

            same_source_excerpt = share_source_excerpt(
                proposal.source_references,
                evidence.source_references,
            )

            if not matched_claims and not same_source_excerpt:
                continue

            findings.append(
                CareerEvidenceOverlapFinding(
                    proposal_id=proposal.proposal_id,
                    scope=(EvidenceOverlapScope.APPROVED_EVIDENCE),
                    matching_evidence_id=(evidence.evidence_id),
                    matched_claims=matched_claims,
                    same_source_excerpt=(same_source_excerpt),
                )
            )

    return findings


def find_matching_claims(
    proposal_claims: list[str],
    comparison_claims: list[str],
) -> list[str]:
    """Return claims that match after deterministic normalisation."""

    comparison_values = {
        normalise_duplicate_text(claim)
        for claim in comparison_claims
        if normalise_duplicate_text(claim)
    }

    return [
        claim
        for claim in proposal_claims
        if (normalise_duplicate_text(claim) in comparison_values)
    ]


def share_source_excerpt(
    left_references: list[SourceReference],
    right_references: list[SourceReference],
) -> bool:
    """Detect identical provenance anchors."""

    left_keys = {
        build_source_key(reference)
        for reference in left_references
        if reference.source_excerpt
    }

    right_keys = {
        build_source_key(reference)
        for reference in right_references
        if reference.source_excerpt
    }

    return bool(left_keys & right_keys)


def build_source_key(
    reference: SourceReference,
) -> tuple[str, str, str]:
    """Build a deterministic provenance-comparison key."""

    return (
        reference.source_type.value,
        reference.source_id,
        normalise_duplicate_text(reference.source_excerpt or ""),
    )


def normalise_duplicate_text(
    text: str,
) -> str:
    """Normalise case, Unicode, punctuation and whitespace."""

    normalised = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    normalised = NON_WORD_PATTERN.sub(
        " ",
        normalised,
    )

    return " ".join(normalised.split())
