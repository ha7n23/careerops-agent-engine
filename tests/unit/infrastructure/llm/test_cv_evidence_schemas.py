"""Tests for bounded provider-facing CV evidence schemas."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.application.ports.cv_evidence_extractor import (
    MAX_CV_EVIDENCE_CANDIDATES,
)
from careerops_agent_engine.domain.enums import EvidenceCategory
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedCareerEvidenceCandidate,
    ExtractedCareerEvidenceSet,
)


def build_candidate() -> ExtractedCareerEvidenceCandidate:
    """Create one valid provider-facing evidence candidate."""

    return ExtractedCareerEvidenceCandidate(
        category=EvidenceCategory.SKILL,
        title="Python",
        source_section_order_index=0,
        source_excerpt="Python",
        technologies=["Python"],
        capabilities=[],
        claims=["Python"],
        warnings=[],
    )


def test_candidate_batch_rejects_more_than_supported_limit() -> None:
    """Provider output cannot exceed the application review limit."""

    with pytest.raises(ValidationError):
        ExtractedCareerEvidenceSet(
            candidates=[
                build_candidate() for _ in range(MAX_CV_EVIDENCE_CANDIDATES + 1)
            ]
        )


def test_candidate_limit_is_exposed_in_json_schema() -> None:
    """The structured-output schema should communicate the limit to Groq."""

    schema = ExtractedCareerEvidenceSet.model_json_schema()

    assert schema["properties"]["candidates"]["maxItems"] == MAX_CV_EVIDENCE_CANDIDATES
