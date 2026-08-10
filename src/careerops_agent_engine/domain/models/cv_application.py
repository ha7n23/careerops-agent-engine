"""Domain models for deterministic application of approved CV changes."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    CVChangeApplicationMode,
    CVSection,
)
from careerops_agent_engine.domain.models.base import (
    DomainModel,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
)


class AppliedCVChange(DomainModel):
    """One approved proposal applied to a precise source location."""

    change_id: str = Field(
        min_length=1,
        max_length=64,
    )

    proposal_id: str = Field(
        min_length=1,
        max_length=64,
    )

    section: CVSection

    application_mode: CVChangeApplicationMode

    source_anchor: str = Field(
        min_length=1,
        max_length=1_500,
    )

    original_text: str = Field(
        min_length=1,
        max_length=1_500,
    )

    applied_text: str = Field(
        min_length=1,
        max_length=1_500,
    )

    anchor_evidence_ids: list[str] = Field(
        default_factory=list,
    )

    requirement_ids: list[str] = Field(
        min_length=1,
    )

    supporting_evidence_ids: list[str] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_change(
        self,
    ) -> Self:
        """Enforce unambiguous and evidence-grounded change provenance."""

        if self.original_text == self.applied_text:
            raise ValueError("An applied CV change must alter the source text.")

        identifier_groups = (
            (
                "Anchor evidence identifiers",
                self.anchor_evidence_ids,
            ),
            (
                "Requirement identifiers",
                self.requirement_ids,
            ),
            (
                "Supporting evidence identifiers",
                self.supporting_evidence_ids,
            ),
        )

        for label, identifiers in identifier_groups:
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{label} must be unique.")

        if not set(self.anchor_evidence_ids) <= set(self.supporting_evidence_ids):
            raise ValueError("Anchor evidence must also be supporting evidence.")

        return self


class StructuredCVTailoringResult(DomainModel):
    """Complete structured CV plus exact changes applied to it."""

    structured_cv: StructuredCV

    applied_changes: list[AppliedCVChange] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_changes(
        self,
    ) -> Self:
        """Require unique applied change and proposal identifiers."""

        change_ids = [change.change_id for change in self.applied_changes]

        proposal_ids = [change.proposal_id for change in self.applied_changes]

        if len(change_ids) != len(set(change_ids)):
            raise ValueError("Applied CV change identifiers must be unique.")

        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("A CV proposal may be applied only once.")

        return self
