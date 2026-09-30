"""Deterministic application of approved CV proposals."""

import re
from dataclasses import dataclass
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    StructuredCVAssemblyError,
)
from careerops_agent_engine.domain.enums import (
    CVChangeApplicationMode,
    CVSection,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
    StructuredCVTailoringResult,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)


@dataclass(frozen=True)
class ResolvedAnchor:
    """One uniquely resolved span inside source CV text."""

    start: int
    end: int
    source_anchor: str
    original_text: str
    anchor_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedProposalChange:
    """One proposal resolved before any source mutation occurs."""

    proposal: CVChangeProposal
    section_index: int | None
    anchor: ResolvedAnchor | None


class StructuredCVProposalApplier:
    """Apply approved proposals through replacement or audited insertion."""

    def apply(
        self,
        *,
        base_cv: StructuredCV,
        proposals: list[CVChangeProposal],
        approved_evidence: list[CareerEvidence],
    ) -> StructuredCVTailoringResult:
        """Apply every proposal without rewriting unrelated CV content."""

        if not proposals:
            raise StructuredCVAssemblyError(
                "At least one final CV proposal is required."
            )

        proposal_ids = [proposal.proposal_id for proposal in proposals]

        if len(proposal_ids) != len(set(proposal_ids)):
            raise StructuredCVAssemblyError(
                "Final CV proposal identifiers must be unique."
            )

        evidence_by_id = self._build_evidence_index(approved_evidence)

        resolved_changes = [
            self._resolve_proposal(
                base_cv=base_cv,
                proposal=proposal,
                evidence_by_id=evidence_by_id,
            )
            for proposal in proposals
        ]

        anchored_changes = [
            change for change in resolved_changes if change.anchor is not None
        ]

        self._validate_non_overlapping_changes(anchored_changes)

        updated_sections = list(base_cv.sections)

        applied_changes: list[AppliedCVChange] = []

        changes_by_section: dict[
            int,
            list[ResolvedProposalChange],
        ] = {}

        for change in resolved_changes:
            if change.anchor is None or change.section_index is None:
                continue

            changes_by_section.setdefault(
                change.section_index,
                [],
            ).append(change)

        for section_index, section_changes in changes_by_section.items():
            section = updated_sections[section_index]

            if section.free_text is None:
                raise StructuredCVAssemblyError(
                    "Anchored proposal application currently "
                    "requires source section free text."
                )

            updated_text = section.free_text

            for change in sorted(
                section_changes,
                key=resolved_anchor_start,
                reverse=True,
            ):
                anchor = change.anchor

                if anchor is None:
                    raise StructuredCVAssemblyError(
                        "An anchored CV change requires a resolved source span."
                    )

                updated_text = (
                    updated_text[: anchor.start]
                    + change.proposal.proposed_text
                    + updated_text[anchor.end :]
                )

                applied_changes.append(
                    AppliedCVChange(
                        change_id=build_applied_change_id(
                            proposal_id=(change.proposal.proposal_id)
                        ),
                        proposal_id=(change.proposal.proposal_id),
                        section=(change.proposal.section),
                        application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                        source_anchor=(anchor.source_anchor),
                        original_text=(anchor.original_text),
                        applied_text=(change.proposal.proposed_text),
                        anchor_evidence_ids=list(anchor.anchor_evidence_ids),
                        requirement_ids=list(change.proposal.requirement_ids),
                        supporting_evidence_ids=list(
                            change.proposal.supporting_evidence_ids
                        ),
                    )
                )

            updated_sections[section_index] = section.model_copy(
                update={
                    "free_text": updated_text,
                }
            )

        for change in resolved_changes:
            if change.anchor is not None:
                continue

            updated_sections = self._insert_evidence_backed_proposal(
                sections=updated_sections,
                proposal=change.proposal,
            )

            applied_changes.append(
                AppliedCVChange(
                    change_id=build_applied_change_id(
                        proposal_id=change.proposal.proposal_id
                    ),
                    proposal_id=change.proposal.proposal_id,
                    section=change.proposal.section,
                    application_mode=(
                        CVChangeApplicationMode.EVIDENCE_BACKED_INSERTION
                    ),
                    source_anchor=None,
                    original_text=None,
                    applied_text=change.proposal.proposed_text,
                    anchor_evidence_ids=[],
                    requirement_ids=list(change.proposal.requirement_ids),
                    supporting_evidence_ids=list(
                        change.proposal.supporting_evidence_ids
                    ),
                )
            )

        applied_by_proposal = {change.proposal_id: change for change in applied_changes}

        ordered_changes = [
            applied_by_proposal[proposal.proposal_id] for proposal in proposals
        ]

        tailored_cv = base_cv.model_copy(
            update={
                "sections": updated_sections,
            }
        )

        return StructuredCVTailoringResult(
            structured_cv=tailored_cv,
            applied_changes=ordered_changes,
        )

    @staticmethod
    def _build_evidence_index(
        approved_evidence: list[CareerEvidence],
    ) -> dict[str, CareerEvidence]:
        """Index only trusted approved evidence."""

        evidence_by_id: dict[
            str,
            CareerEvidence,
        ] = {}

        for evidence in approved_evidence:
            if evidence.verification_status is not VerificationStatus.APPROVED:
                raise StructuredCVAssemblyError(
                    "CV tailoring may use only approved evidence."
                )

            if evidence.evidence_id in evidence_by_id:
                raise StructuredCVAssemblyError(
                    "Approved evidence identifiers must be unique."
                )

            evidence_by_id[evidence.evidence_id] = evidence

        return evidence_by_id

    def _resolve_proposal(
        self,
        *,
        base_cv: StructuredCV,
        proposal: CVChangeProposal,
        evidence_by_id: dict[str, CareerEvidence],
    ) -> ResolvedProposalChange:
        """Resolve one proposal to replacement or insertion."""

        supporting_evidence = []

        for evidence_id in proposal.supporting_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)

            if evidence is None:
                raise StructuredCVAssemblyError(
                    "Every proposal supporting evidence "
                    "identifier must reference approved evidence."
                )

            supporting_evidence.append(evidence)

        if proposal.target_entry_id is not None:
            raise StructuredCVAssemblyError(
                "Entry-targeted CV proposals are not supported "
                "until structured source entries are available."
            )

        matching_sections = [
            (
                index,
                section,
            )
            for index, section in enumerate(base_cv.sections)
            if section.section is proposal.section
        ]

        if len(matching_sections) > 1:
            raise StructuredCVAssemblyError(
                "A CV proposal must target exactly one structured CV section."
            )

        if not matching_sections:
            return self._resolve_missing_section_proposal(
                base_cv=base_cv,
                proposal=proposal,
                supporting_evidence=supporting_evidence,
            )

        section_index, section = matching_sections[0]

        if section.free_text is None and proposal.current_text is not None:
            raise StructuredCVAssemblyError(
                "Anchored proposal application currently "
                "requires source section free text."
            )

        if proposal.current_text is not None and section.free_text is not None:
            anchor = self._resolve_unique_anchor(
                source_text=section.free_text,
                anchor_text=proposal.current_text,
                anchor_evidence_ids=(),
            )

            return ResolvedProposalChange(
                proposal=proposal,
                section_index=section_index,
                anchor=anchor,
            )

        candidate_anchors: dict[
            str,
            set[str],
        ] = {}

        if section.free_text is not None:
            for evidence in supporting_evidence:
                for reference in evidence.source_references:
                    if (
                        reference.source_type is not EvidenceSourceType.UPLOADED_CV
                        or reference.source_id != base_cv.source_document_id
                        or reference.source_excerpt is None
                    ):
                        continue

                    if self._has_anchor(
                        source_text=section.free_text,
                        anchor_text=(reference.source_excerpt),
                    ):
                        candidate_anchors.setdefault(
                            reference.source_excerpt,
                            set(),
                        ).add(evidence.evidence_id)

        if len(candidate_anchors) > 1:
            raise StructuredCVAssemblyError(
                "A CV proposal without current text requires "
                "exactly one grounded source anchor in its "
                "target section."
            )

        if not candidate_anchors:
            return ResolvedProposalChange(
                proposal=proposal,
                section_index=section_index,
                anchor=None,
            )

        if section.free_text is None:
            raise StructuredCVAssemblyError(
                "A grounded source anchor requires source section free text."
            )

        anchor_text, evidence_ids = next(iter(candidate_anchors.items()))

        anchor = self._resolve_unique_anchor(
            source_text=section.free_text,
            anchor_text=anchor_text,
            anchor_evidence_ids=tuple(sorted(evidence_ids)),
        )

        return ResolvedProposalChange(
            proposal=proposal,
            section_index=section_index,
            anchor=anchor,
        )

    @staticmethod
    def _resolve_missing_section_proposal(
        *,
        base_cv: StructuredCV,
        proposal: CVChangeProposal,
        supporting_evidence: list[CareerEvidence],
    ) -> ResolvedProposalChange:
        """Reconcile one source anchor or select missing-section insertion."""

        candidate_anchors: dict[str, set[str]] = {}

        if proposal.current_text is not None:
            candidate_anchors[proposal.current_text] = set()
        else:
            for evidence in supporting_evidence:
                for reference in evidence.source_references:
                    if (
                        reference.source_type is EvidenceSourceType.UPLOADED_CV
                        and reference.source_id == base_cv.source_document_id
                        and reference.source_excerpt is not None
                    ):
                        candidate_anchors.setdefault(
                            reference.source_excerpt,
                            set(),
                        ).add(evidence.evidence_id)

        resolved_candidates: list[tuple[int, CVChangeProposal, ResolvedAnchor]] = []

        for section_index, section in enumerate(base_cv.sections):
            if section.free_text is None:
                continue

            for anchor_text, evidence_ids in candidate_anchors.items():
                matches = list(
                    build_whitespace_pattern(anchor_text).finditer(section.free_text)
                )

                for match in matches:
                    resolved_candidates.append(
                        (
                            section_index,
                            proposal.model_copy(
                                update={"section": section.section},
                            ),
                            ResolvedAnchor(
                                start=match.start(),
                                end=match.end(),
                                source_anchor=anchor_text,
                                original_text=match.group(0),
                                anchor_evidence_ids=tuple(sorted(evidence_ids)),
                            ),
                        )
                    )

        if len(resolved_candidates) > 1:
            raise StructuredCVAssemblyError(
                "A CV proposal whose declared section is absent requires "
                "exactly one grounded source anchor across the selected CV."
            )

        if not resolved_candidates:
            if proposal.current_text is not None:
                raise StructuredCVAssemblyError(
                    "An explicit CV source anchor must match exactly once "
                    "across the selected CV."
                )

            return ResolvedProposalChange(
                proposal=proposal,
                section_index=None,
                anchor=None,
            )

        section_index, reconciled_proposal, anchor = resolved_candidates[0]

        return ResolvedProposalChange(
            proposal=reconciled_proposal,
            section_index=section_index,
            anchor=anchor,
        )

    @staticmethod
    def _insert_evidence_backed_proposal(
        *,
        sections: list[StructuredCVSection],
        proposal: CVChangeProposal,
    ) -> list[StructuredCVSection]:
        """Insert exact approved wording without claiming a source replacement."""

        updated_sections = list(sections)
        inserted_line = f"- {proposal.proposed_text}"

        for index, section in enumerate(updated_sections):
            if section.section is not proposal.section:
                continue

            if section.free_text is None:
                updated_text = inserted_line
            else:
                separator = "" if section.free_text.endswith("\n") else "\n"
                updated_text = f"{section.free_text}{separator}{inserted_line}"

            updated_sections[index] = section.model_copy(
                update={"free_text": updated_text}
            )

            return updated_sections

        updated_sections.append(
            StructuredCVSection(
                section=proposal.section,
                heading=build_section_heading(proposal.section),
                free_text=inserted_line,
            )
        )

        return updated_sections

    @staticmethod
    def _has_anchor(
        *,
        source_text: str,
        anchor_text: str,
    ) -> bool:
        """Return whether whitespace-normalised anchor text occurs."""

        return bool(build_whitespace_pattern(anchor_text).search(source_text))

    @staticmethod
    def _resolve_unique_anchor(
        *,
        source_text: str,
        anchor_text: str,
        anchor_evidence_ids: tuple[str, ...],
    ) -> ResolvedAnchor:
        """Resolve exactly one whitespace-tolerant source span."""

        matches = list(build_whitespace_pattern(anchor_text).finditer(source_text))

        if len(matches) != 1:
            raise StructuredCVAssemblyError(
                "A CV change source anchor must match "
                "exactly once in its target section."
            )

        match = matches[0]

        return ResolvedAnchor(
            start=match.start(),
            end=match.end(),
            source_anchor=anchor_text,
            original_text=match.group(0),
            anchor_evidence_ids=(anchor_evidence_ids),
        )

    @staticmethod
    def _validate_non_overlapping_changes(
        changes: list[ResolvedProposalChange],
    ) -> None:
        """Prevent two proposals from mutating the same source span."""

        by_section: dict[
            int,
            list[ResolvedProposalChange],
        ] = {}

        for change in changes:
            if change.section_index is None or change.anchor is None:
                continue

            by_section.setdefault(
                change.section_index,
                [],
            ).append(change)

        for section_changes in by_section.values():
            ordered = sorted(
                section_changes,
                key=resolved_anchor_bounds,
            )

            for previous, current in zip(
                ordered,
                ordered[1:],
                strict=False,
            ):
                previous_anchor = previous.anchor
                current_anchor = current.anchor

                if previous_anchor is None or current_anchor is None:
                    continue

                if current_anchor.start < previous_anchor.end:
                    raise StructuredCVAssemblyError(
                        "CV proposals cannot modify overlapping source spans."
                    )


def resolved_anchor_start(
    change: ResolvedProposalChange,
) -> int:
    """Return a resolved anchor start for deterministic sorting."""

    if change.anchor is None:
        raise StructuredCVAssemblyError(
            "An anchored CV change requires a resolved source span."
        )

    return change.anchor.start


def resolved_anchor_bounds(
    change: ResolvedProposalChange,
) -> tuple[int, int]:
    """Return resolved anchor bounds for overlap validation."""

    if change.anchor is None:
        raise StructuredCVAssemblyError(
            "An anchored CV change requires a resolved source span."
        )

    return (
        change.anchor.start,
        change.anchor.end,
    )


def build_section_heading(
    section: CVSection,
) -> str:
    """Build a stable human-readable heading for a missing CV section."""

    return section.value.replace("_", " ").title()


def build_whitespace_pattern(
    text: str,
) -> re.Pattern[str]:
    """Build a regex matching the same text across whitespace differences."""

    tokens = re.findall(
        r"\S+",
        text,
    )

    if not tokens:
        raise StructuredCVAssemblyError("A CV change source anchor cannot be blank.")

    pattern = r"\s+".join(re.escape(token) for token in tokens)

    return re.compile(
        pattern,
        flags=re.MULTILINE,
    )


def build_applied_change_id(
    *,
    proposal_id: str,
) -> str:
    """Build a stable applied-change identifier."""

    digest = sha256(proposal_id.encode("utf-8")).hexdigest()[:16].upper()

    return f"CHG-{digest}"
