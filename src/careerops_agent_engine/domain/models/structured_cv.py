"""Domain models for complete structured CV content."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    CVSection,
)
from careerops_agent_engine.domain.models.base import (
    DomainModel,
)


class CVBullet(DomainModel):
    """One evidence-grounded bullet inside a CV entry."""

    bullet_id: str = Field(
        min_length=1,
        max_length=64,
    )

    text: str = Field(
        min_length=1,
        max_length=1_500,
    )

    supporting_evidence_ids: list[str] = Field(
        min_length=1,
    )

    requirement_ids: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_identifiers(
        self,
    ) -> Self:
        """Reject duplicated provenance identifiers."""

        if len(self.supporting_evidence_ids) != len(set(self.supporting_evidence_ids)):
            raise ValueError("Supporting evidence identifiers must be unique.")

        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("Requirement identifiers must be unique.")

        return self


class CVEntry(DomainModel):
    """One structured entry within a CV section."""

    entry_id: str = Field(
        min_length=1,
        max_length=64,
    )

    section: CVSection

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    subtitle: str | None = Field(
        default=None,
        max_length=250,
    )

    location: str | None = Field(
        default=None,
        max_length=250,
    )

    date_text: str | None = Field(
        default=None,
        max_length=120,
    )

    bullets: list[CVBullet] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_bullets(
        self,
    ) -> Self:
        """Ensure bullet identifiers are unique per entry."""

        bullet_ids = [bullet.bullet_id for bullet in self.bullets]

        if len(bullet_ids) != len(set(bullet_ids)):
            raise ValueError("CV bullet identifiers must be unique within an entry.")

        return self


class StructuredCVSection(DomainModel):
    """One complete renderable CV section."""

    section: CVSection

    heading: str = Field(
        min_length=1,
        max_length=120,
    )

    entries: list[CVEntry] = Field(
        default_factory=list,
    )

    free_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=20_000,
    )

    @model_validator(mode="after")
    def validate_content(
        self,
    ) -> Self:
        """Require section content and consistent entry types."""

        if not self.entries and self.free_text is None:
            raise ValueError(
                "A structured CV section must contain entries or free text."
            )

        if any(entry.section is not self.section for entry in self.entries):
            raise ValueError("Every CV entry must belong to its containing section.")

        entry_ids = [entry.entry_id for entry in self.entries]

        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("CV entry identifiers must be unique within a section.")

        return self


class StructuredCV(DomainModel):
    """Complete CV content before document rendering."""

    cv_id: str = Field(
        min_length=1,
        max_length=64,
    )

    source_document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    preamble_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=4_000,
    )

    sections: list[StructuredCVSection] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_sections(
        self,
    ) -> Self:
        """Reject duplicate section types and global entry IDs."""

        section_types = [section.section for section in self.sections]

        if len(section_types) != len(set(section_types)):
            raise ValueError("A structured CV may contain each section type only once.")

        entry_ids = [
            entry.entry_id for section in self.sections for entry in section.entries
        ]

        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("CV entry identifiers must be globally unique.")

        bullet_ids = [
            bullet.bullet_id
            for section in self.sections
            for entry in section.entries
            for bullet in entry.bullets
        ]

        if len(bullet_ids) != len(set(bullet_ids)):
            raise ValueError("CV bullet identifiers must be globally unique.")

        return self
