"""Live integration test for grounded CV proposal generation."""

import os

import pytest

from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_cv_proposal_generator,
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


def test_live_grounded_cv_proposal() -> None:
    """Generate one real proposal from approved PostgreSQL evidence."""

    settings = get_settings()

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    repository = SqlAlchemyEvidenceRepository(session_factory)

    try:
        # Verify our expected seeded evidence exists before making
        # an external model call.
        careerops_evidence = repository.get_approved(
            user_id=DEVELOPMENT_USER_ID,
            evidence_id="EVD-DEMO-CAREEROPS",
        )
        aws_evidence = repository.get_approved(
            user_id=DEVELOPMENT_USER_ID,
            evidence_id="EVD-DEMO-AWS",
        )

        assert careerops_evidence is not None
        assert aws_evidence is not None

        requirement = JobRequirement(
            requirement_id="REQ-LIVE-PYTHON",
            name="Python Software Engineering",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected=(
                "Demonstrated practical Python software-engineering experience."
            ),
            importance_score=5,
            source_text=("Strong Python software-engineering experience is required."),
        )

        evidence_match = EvidenceMatch(
            requirement_id=requirement.requirement_id,
            match_strength=MatchStrength.STRONG,
            direct_evidence_ids=[
                "EVD-DEMO-CAREEROPS",
                "EVD-DEMO-AWS",
            ],
            related_evidence_ids=[],
            explanation=(
                "Approved project evidence directly demonstrates "
                "Python application development."
            ),
            gap=False,
        )

        service = CVProposalGenerationService(
            repository=repository,
            generator=create_cv_proposal_generator(settings),
        )

        proposal = service.generate_for_requirement(
            job_id="JOB-LIVE-CV-PROPOSAL-001",
            user_id=DEVELOPMENT_USER_ID,
            requirement=requirement,
            evidence_match=evidence_match,
        )

        assert proposal is not None

        print()
        print("Live CV proposal:")
        print(proposal.model_dump_json(indent=2))

        assert proposal.requirement_ids == [requirement.requirement_id]

        assert set(proposal.supporting_evidence_ids).issubset(
            {
                "EVD-DEMO-CAREEROPS",
                "EVD-DEMO-AWS",
            }
        )

        assert proposal.supporting_evidence_ids
        assert proposal.target_entry_id is None
        assert proposal.current_text is None
        assert proposal.requires_human_approval is True
        assert proposal.proposed_text.strip()

    finally:
        engine.dispose()
