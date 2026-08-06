"""Live tests for Gemini requirement extraction."""

import os

import pytest

from careerops_agent_engine.infrastructure.llm.factory import (
    create_requirement_extractor,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Live LLM tests are disabled.",
)
def test_google_extractor_returns_structured_requirements() -> None:
    """Gemini should return validated requirements for synthetic input."""

    extractor = create_requirement_extractor()

    extraction = extractor.extract(
        """
        We are hiring a Junior AI Engineer.

        Essential requirements:
        - Strong Python development skills.
        - Experience developing FastAPI services.

        Desirable requirements:
        - Familiarity with LangGraph.
        - Exposure to AWS cloud deployment.
        """,
        job_id="JOB-INTEGRATION-001",
    )

    assert extraction.requirements
    assert extraction.requirements[0].requirement_id == "REQ-001"

    names = {requirement.name.lower() for requirement in extraction.requirements}

    assert any("python" in name for name in names)
    assert any("fastapi" in name for name in names)
