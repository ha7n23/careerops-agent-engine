"""Live integration test for batched CV claim verification."""

import json
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


def test_live_batched_verifier_separates_supported_and_unsupported() -> None:
    """Real batch verification should assess proposals independently."""

    settings = get_settings()

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    repository = SqlAlchemyEvidenceRepository(session_factory)

    try:
        supported_proposal = CVChangeProposal(
            proposal_id="CVP-LIVE-VERIFY-SUPPORTED",
            section=CVSection.PROJECTS,
            proposed_text=(
                "Built a stateful LangGraph workflow with structured Gemini extraction."
            ),
            requirement_ids=["REQ-LIVE-SUPPORTED"],
            supporting_evidence_ids=["EVD-DEMO-CAREEROPS"],
            confidence_score=0.95,
            warnings=[],
        )
        unsupported_proposal = CVChangeProposal(
            proposal_id="CVP-LIVE-VERIFY-UNSUPPORTED",
            section=CVSection.PROJECTS,
            proposed_text=(
                "Built a stateful LangGraph workflow with "
                "structured Gemini extraction and reduced "
                "production latency by 40 percent."
            ),
            requirement_ids=["REQ-LIVE-UNSUPPORTED"],
            supporting_evidence_ids=["EVD-DEMO-CAREEROPS"],
            confidence_score=0.9,
            warnings=[],
        )

        service = CVClaimVerificationService(
            repository=repository,
            verifier=create_cv_claim_verifier(settings),
        )

        reports = service.verify_proposals(
            user_id=DEVELOPMENT_USER_ID,
            proposals=[
                supported_proposal,
                unsupported_proposal,
            ],
        )

        print()
        print("Live batched claim-verification reports:")
        print(
            json.dumps(
                [report.model_dump(mode="json") for report in reports],
                indent=2,
            )
        )

        assert len(reports) == 2

        reports_by_proposal = {report.proposal_id: report for report in reports}

        supported_report = reports_by_proposal["CVP-LIVE-VERIFY-SUPPORTED"]
        unsupported_report = reports_by_proposal["CVP-LIVE-VERIFY-UNSUPPORTED"]

        assert supported_report.coverage_complete is True
        assert supported_report.fully_supported is True
        assert supported_report.unsupported_claims == []

        assert unsupported_report.coverage_complete is True
        assert unsupported_report.fully_supported is False
        assert unsupported_report.unsupported_claims

        combined_unsupported = " ".join(
            unsupported_report.unsupported_claims
        ).casefold()

        assert "40" in combined_unsupported
        assert "latency" in combined_unsupported

    finally:
        engine.dispose()
