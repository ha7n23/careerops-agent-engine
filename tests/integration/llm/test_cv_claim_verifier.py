"""Live integration test for strict CV claim verification."""

import os

import pytest

from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.domain.enums import CVSection
from careerops_agent_engine.domain.models.cv import CVChangeProposal
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_cv_claim_verifier,
)
from careerops_agent_engine.infrastructure.repositories.development_evidence import (
    DEVELOPMENT_USER_ID,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Live LLM tests are disabled.",
)


def test_live_verifier_blocks_invented_metric() -> None:
    """Real verification should identify an unsupported metric."""

    settings = get_settings()

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    repository = SqlAlchemyEvidenceRepository(session_factory)

    try:
        proposal = CVChangeProposal(
            proposal_id="CVP-LIVE-VERIFY-001",
            section=CVSection.PROJECTS,
            proposed_text=(
                "Built a stateful LangGraph workflow with "
                "structured Gemini extraction and reduced "
                "production latency by 40 percent."
            ),
            requirement_ids=["REQ-LIVE-001"],
            supporting_evidence_ids=["EVD-DEMO-CAREEROPS"],
            confidence_score=0.9,
            warnings=[],
        )

        service = CVClaimVerificationService(
            repository=repository,
            verifier=create_cv_claim_verifier(settings),
        )

        report = service.verify_proposal(
            user_id=DEVELOPMENT_USER_ID,
            proposal=proposal,
        )

        print()
        print("Live claim-verification report:")
        print(report.model_dump_json(indent=2))

        assert report.coverage_complete is True
        assert report.fully_supported is False
        assert report.unsupported_claims

        combined_unsupported = " ".join(report.unsupported_claims).casefold()

        assert "40" in combined_unsupported
        assert "latency" in combined_unsupported

    finally:
        engine.dispose()
