"""Tests for the evidence-grounded job-analysis graph."""

from careerops_agent_engine.agents.graphs.job_analysis import (
    build_job_analysis_graph,
)
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)


class FakeRequirementExtractor:
    """Deterministic replacement for external requirement extraction."""

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Return predictable requirements without an API call."""

        del job_description, job_id

        return JobRequirementExtraction(
            role_title="Junior AI Engineer",
            requirements=[
                JobRequirement(
                    requirement_id="REQ-PYTHON",
                    name="Python",
                    category=RequirementCategory.ESSENTIAL,
                    evidence_expected=(
                        "Practical Python software-engineering experience."
                    ),
                    importance_score=5,
                    source_text=("Strong Python development experience is required."),
                ),
                JobRequirement(
                    requirement_id="REQ-LANGGRAPH",
                    name="LangGraph",
                    category=RequirementCategory.ESSENTIAL,
                    evidence_expected=("Implementation of stateful agent workflows."),
                    importance_score=4,
                    source_text="Experience with LangGraph is required.",
                ),
            ],
        )


class FakeEvidenceDiscoveryRunner:
    """Return deterministic evidence matches without calling an LLM."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return one validated match for the supplied requirement."""

        assert user_id == "USER-TEST-001"

        if requirement.requirement_id == "REQ-PYTHON":
            return EvidenceMatch(
                requirement_id=requirement.requirement_id,
                match_strength=MatchStrength.STRONG,
                direct_evidence_ids=["EVD-PYTHON"],
                related_evidence_ids=[],
                explanation=(
                    "Approved evidence directly demonstrates "
                    "Python software-engineering experience."
                ),
                gap=False,
            )

        return EvidenceMatch(
            requirement_id=requirement.requirement_id,
            match_strength=MatchStrength.NONE,
            direct_evidence_ids=[],
            related_evidence_ids=[],
            explanation=("No approved LangGraph evidence was found."),
            gap=True,
        )


def build_test_graph():
    """Create a graph using deterministic fake adapters."""

    return build_job_analysis_graph(
        requirement_extractor=FakeRequirementExtractor(),
        evidence_discovery_runner=FakeEvidenceDiscoveryRunner(),
    )


def test_valid_job_runs_evidence_grounded_analysis_path() -> None:
    """Valid input should extract, match and score requirements."""

    graph = build_test_graph()

    result = graph.invoke(
        {
            "job_id": "JOB-001",
            "user_id": "USER-TEST-001",
            "job_description": (
                "We require strong Python development and LangGraph "
                "workflow experience."
            ),
            "audit_events": [],
        }
    )

    assert result["status"] == "completed"
    assert result["validation_error"] is None
    assert result["role_title"] == "Junior AI Engineer"

    assert len(result["requirements"]) == 2
    assert len(result["evidence_matches"]) == 2
    assert result["fit_score"] == 55.56

    python_match = EvidenceMatch.model_validate(result["evidence_matches"][0])
    langgraph_match = EvidenceMatch.model_validate(result["evidence_matches"][1])

    assert python_match.match_strength is MatchStrength.STRONG
    assert python_match.direct_evidence_ids == ["EVD-PYTHON"]
    assert python_match.gap is False

    assert langgraph_match.match_strength is MatchStrength.NONE
    assert langgraph_match.direct_evidence_ids == []
    assert langgraph_match.gap is True

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "evidence_discovery_completed",
        "fit_score_calculated",
        "job_analysis_completed",
    ]


def test_invalid_job_skips_extraction_and_evidence_discovery() -> None:
    """Invalid input should route directly to rejection."""

    graph = build_test_graph()

    result = graph.invoke(
        {
            "job_id": "JOB-002",
            "user_id": "USER-TEST-001",
            "job_description": "Too short",
            "audit_events": [],
        }
    )

    assert result["status"] == "invalid"
    assert result["validation_error"] is not None
    assert "requirements" not in result
    assert "evidence_matches" not in result
    assert "fit_score" not in result

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_invalid",
        "job_analysis_rejected",
    ]
