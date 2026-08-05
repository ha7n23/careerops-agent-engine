"""Domain models for evidence-grounded CV proposals."""

from typing import Literal, Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import CVSection
from careerops_agent_engine.domain.models.base import DomainModel


class CVChangeProposal(DomainModel):
    """One proposed evidence-grounded change to a CV."""

    proposal_id: str = Field(min_length=1, max_length=64)
    section: CVSection
    target_entry_id: str | None = Field(default=None, max_length=64)

    current_text: str | None = Field(default=None, max_length=1_500)
    proposed_text: str = Field(min_length=1, max_length=1_500)

    requirement_ids: list[str] = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(min_length=1)

    confidence_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    requires_human_approval: Literal[True] = True

    @model_validator(mode="after")
    def validate_proposal_integrity(self) -> Self:
        """Reject ambiguous, duplicated or ineffective proposals."""

        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("Requirement identifiers must be unique.")

        if len(self.supporting_evidence_ids) != len(set(self.supporting_evidence_ids)):
            raise ValueError("Supporting evidence identifiers must be unique.")

        if self.current_text is not None and self.current_text == self.proposed_text:
            raise ValueError("Proposed text must differ from the current text.")

        return self
