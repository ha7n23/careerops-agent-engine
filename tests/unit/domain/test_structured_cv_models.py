"""Tests for render-independent structured CV models."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import (
    CVSection,
)
from careerops_agent_engine.domain.models.structured_cv import (
    CVBullet,
    CVEntry,
    StructuredCV,
    StructuredCVSection,
)


def build_bullet(
    *,
    bullet_id: str = "BLT-001",
) -> CVBullet:
    """Create one grounded CV bullet."""

    return CVBullet(
        bullet_id=bullet_id,
        text=("Built a production-style Python API using FastAPI."),
        supporting_evidence_ids=[
            "EVD-001",
        ],
        requirement_ids=[
            "REQ-001",
        ],
    )


def build_entry(
    *,
    entry_id: str = "ENT-001",
    bullet_id: str = "BLT-001",
) -> CVEntry:
    """Create one project entry."""

    return CVEntry(
        entry_id=entry_id,
        section=CVSection.PROJECTS,
        title="CareerOps Agent Engine",
        subtitle="AI Engineering Project",
        bullets=[
            build_bullet(
                bullet_id=bullet_id,
            )
        ],
    )


def test_structured_cv_accepts_grounded_content() -> None:
    """A complete structured CV should validate."""

    cv = StructuredCV(
        cv_id="CV-001",
        source_document_id="DOC-001",
        sections=[
            StructuredCVSection(
                section=CVSection.PROFILE,
                heading="Profile",
                free_text=("AI engineer building grounded LLM applications."),
            ),
            StructuredCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                entries=[build_entry()],
            ),
        ],
    )

    assert cv.cv_id == "CV-001"
    assert len(cv.sections) == 2


def test_section_rejects_entry_from_wrong_section() -> None:
    """An entry cannot silently move across CV sections."""

    with pytest.raises(
        ValidationError,
        match="containing section",
    ):
        StructuredCVSection(
            section=CVSection.EXPERIENCE,
            heading="Experience",
            entries=[build_entry()],
        )


def test_structured_cv_rejects_duplicate_sections() -> None:
    """Each supported section may appear only once."""

    with pytest.raises(
        ValidationError,
        match="section type only once",
    ):
        StructuredCV(
            cv_id="CV-001",
            source_document_id="DOC-001",
            sections=[
                StructuredCVSection(
                    section=CVSection.PROFILE,
                    heading="Profile",
                    free_text="Profile one.",
                ),
                StructuredCVSection(
                    section=CVSection.PROFILE,
                    heading="Summary",
                    free_text="Profile two.",
                ),
            ],
        )


def test_structured_cv_rejects_duplicate_entry_ids() -> None:
    """Entry identifiers must remain globally unambiguous."""

    with pytest.raises(
        ValidationError,
        match="globally unique",
    ):
        StructuredCV(
            cv_id="CV-001",
            source_document_id="DOC-001",
            sections=[
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    entries=[
                        build_entry(
                            entry_id="ENT-DUPLICATE",
                        )
                    ],
                ),
                StructuredCVSection(
                    section=CVSection.EXPERIENCE,
                    heading="Experience",
                    entries=[
                        CVEntry(
                            entry_id="ENT-DUPLICATE",
                            section=(CVSection.EXPERIENCE),
                            title="Software Engineer",
                            bullets=[
                                build_bullet(
                                    bullet_id="BLT-002",
                                )
                            ],
                        )
                    ],
                ),
            ],
        )


def test_structured_cv_rejects_duplicate_bullet_ids() -> None:
    """Bullet identifiers must also remain globally unambiguous."""

    with pytest.raises(
        ValidationError,
        match="globally unique",
    ):
        StructuredCV(
            cv_id="CV-001",
            source_document_id="DOC-001",
            sections=[
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    entries=[
                        build_entry(
                            entry_id="ENT-001",
                            bullet_id="BLT-DUPLICATE",
                        ),
                        build_entry(
                            entry_id="ENT-002",
                            bullet_id="BLT-DUPLICATE",
                        ),
                    ],
                )
            ],
        )


def test_bullet_rejects_duplicate_evidence_ids() -> None:
    """One bullet cannot carry ambiguous duplicate provenance."""

    with pytest.raises(
        ValidationError,
        match="Supporting evidence identifiers",
    ):
        CVBullet(
            bullet_id="BLT-001",
            text="Built a Python API.",
            supporting_evidence_ids=[
                "EVD-001",
                "EVD-001",
            ],
        )
