"""Tests for the job-analysis LangGraph workflow."""

from careerops_agent_engine.agents.graphs.job_analysis import (
    build_job_analysis_graph,
)
from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)


class FakeRequirementExtractor:
    """Deterministic replacement for the external model in graph tests."""

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


def build_test_graph():
    """Create a graph using the deterministic fake extractor."""

    return build_job_analysis_graph(FakeRequirementExtractor())


def test_valid_job_runs_complete_analysis_path() -> None:
    """Valid input should extract requirements and calculate fit."""

    graph = build_test_graph()

    result = graph.invoke(
        {
            "job_id": "JOB-001",
            "job_description": (
                "We require strong Python development and LangGraph "
                "workflow experience."
            ),
            "matched_requirement_ids": ["REQ-PYTHON"],
            "audit_events": [],
        }
    )

    assert result["status"] == "completed"
    assert result["validation_error"] is None
    assert len(result["requirements"]) == 2
    assert result["fit_score"] == 55.56

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "fit_score_calculated",
        "job_analysis_completed",
    ]


def test_invalid_job_skips_extraction_and_scoring() -> None:
    """Invalid input should route directly to rejection."""

    graph = build_test_graph()

    result = graph.invoke(
        {
            "job_id": "JOB-002",
            "job_description": "Too short",
            "matched_requirement_ids": [],
            "audit_events": [],
        }
    )

    assert result["status"] == "invalid"
    assert result["validation_error"] is not None
    assert "requirements" not in result
    assert "fit_score" not in result

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_invalid",
        "job_analysis_rejected",
    ]
