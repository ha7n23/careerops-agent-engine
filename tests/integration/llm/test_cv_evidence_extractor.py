"""Live LLM tests for provider-neutral CV evidence extraction."""

import os

import pytest

from careerops_agent_engine.domain.enums import CVSection
from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
    ParsedCVSection,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_cv_evidence_extractor,
)

pytestmark = pytest.mark.integration


def _normalise(value: str) -> str:
    """Normalise whitespace for grounding comparisons."""

    return " ".join(value.split())


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Live LLM tests are disabled.",
)
def test_cv_evidence_extractor_returns_grounded_candidates() -> None:
    """The configured fast model should extract only source-grounded evidence."""

    document = ParsedCVDocument(
        document_id="DOC-INTEGRATION-001",
        sections=[
            ParsedCVSection(
                section=CVSection.EXPERIENCE,
                heading="Experience",
                text=(
                    "Software Engineer at Example Ltd.\n"
                    "Built Python APIs using FastAPI and Docker."
                ),
                order_index=0,
            ),
        ],
    )

    extractor = create_cv_evidence_extractor()

    candidates = extractor.extract(document=document)

    assert candidates

    source_sections = {
        section.order_index: _normalise(section.text) for section in document.sections
    }

    for candidate in candidates:
        source_text = source_sections[candidate.source_section_order_index]
        source_excerpt = _normalise(candidate.source_excerpt)

        assert source_excerpt in source_text
        assert candidate.claims

        for claim in candidate.claims:
            assert _normalise(claim) in source_excerpt

        for technology in candidate.technologies:
            assert technology.casefold() in source_excerpt.casefold()

    technologies = {
        technology.casefold()
        for candidate in candidates
        for technology in candidate.technologies
    }

    expected_technologies = {
        "python",
        "fastapi",
        "docker",
    }

    assert technologies
    assert technologies <= expected_technologies
    assert technologies & expected_technologies
