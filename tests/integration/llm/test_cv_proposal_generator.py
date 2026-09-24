"""Live integration test for batched grounded CV proposal generation."""

import json
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


def test_live_batched_grounded_cv_proposals() -> None:
    """Generate two grounded proposals through one real batch call."""

    settings = get_settings()

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    repository = SqlAlchemyEvidenceRepository(session_factory)

    try:
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

        python_requirement = JobRequirement(
            requirement_id="REQ-LIVE-PYTHON",
            name="Python Software Engineering",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected=(
                "Demonstrated practical Python software-engineering experience."
            ),
            importance_score=5,
            source_text=("Strong Python software-engineering experience is required."),
        )
        aws_requirement = JobRequirement(
            requirement_id="REQ-LIVE-AWS",
            name="AWS Deployment",
            category=RequirementCategory.DESIRABLE,
            evidence_expected=("Experience deploying applications to AWS."),
            importance_score=3,
            source_text="AWS deployment experience is desirable.",
        )

        python_match = EvidenceMatch(
            requirement_id=python_requirement.requirement_id,
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
        aws_match = EvidenceMatch(
            requirement_id=aws_requirement.requirement_id,
            match_strength=MatchStrength.STRONG,
            direct_evidence_ids=[
                "EVD-DEMO-AWS",
            ],
            related_evidence_ids=[],
            explanation=(
                "Approved AWS evidence directly demonstrates application deployment."
            ),
            gap=False,
        )

        service = CVProposalGenerationService(
            repository=repository,
            generator=create_cv_proposal_generator(settings),
        )

        proposals = service.generate_for_requirements(
            job_id="JOB-LIVE-CV-PROPOSAL-BATCH-001",
            user_id=DEVELOPMENT_USER_ID,
            requirements=[
                python_requirement,
                aws_requirement,
            ],
            evidence_matches=[
                python_match,
                aws_match,
            ],
        )

        print()
        print("Live batched CV proposals:")
        print(
            json.dumps(
                [proposal.model_dump(mode="json") for proposal in proposals],
                indent=2,
            )
        )

        assert len(proposals) == 2

        proposals_by_requirement = {
            proposal.requirement_ids[0]: proposal for proposal in proposals
        }

        assert set(proposals_by_requirement) == {
            "REQ-LIVE-PYTHON",
            "REQ-LIVE-AWS",
        }

        assert set(
            proposals_by_requirement["REQ-LIVE-PYTHON"].supporting_evidence_ids
        ).issubset(
            {
                "EVD-DEMO-CAREEROPS",
                "EVD-DEMO-AWS",
            }
        )
        assert set(
            proposals_by_requirement["REQ-LIVE-AWS"].supporting_evidence_ids
        ).issubset(
            {
                "EVD-DEMO-AWS",
            }
        )

        for proposal in proposals:
            assert proposal.supporting_evidence_ids
            assert proposal.target_entry_id is None
            assert proposal.current_text is None
            assert proposal.requires_human_approval is True
            assert proposal.proposed_text.strip()

    finally:
        engine.dispose()
