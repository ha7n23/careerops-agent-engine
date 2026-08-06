"""Construction of the initial deterministic job-analysis graph."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from careerops_agent_engine.agents.nodes.job_analysis import (
    calculate_mock_fit,
    complete_analysis,
    create_mock_requirements,
    mark_invalid,
    validate_job_input,
)
from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
)


def route_after_validation(
    state: JobAnalysisState,
) -> Literal["valid", "invalid"]:
    """Select the valid or invalid execution branch."""

    if state.get("validation_error") is not None:
        return "invalid"

    return "valid"


def build_job_analysis_graph() -> CompiledStateGraph[
    JobAnalysisState,
    None,
    JobAnalysisState,
    JobAnalysisState,
]:
    """Compile the first deterministic CareerOps workflow."""

    builder = StateGraph(JobAnalysisState)

    builder.add_node("validate_job_input", validate_job_input)
    builder.add_node(
        "create_mock_requirements",
        create_mock_requirements,
    )
    builder.add_node("calculate_mock_fit", calculate_mock_fit)
    builder.add_node("complete_analysis", complete_analysis)
    builder.add_node("mark_invalid", mark_invalid)

    builder.add_edge(START, "validate_job_input")

    builder.add_conditional_edges(
        "validate_job_input",
        route_after_validation,
        {
            "valid": "create_mock_requirements",
            "invalid": "mark_invalid",
        },
    )

    builder.add_edge(
        "create_mock_requirements",
        "calculate_mock_fit",
    )
    builder.add_edge("calculate_mock_fit", "complete_analysis")
    builder.add_edge("complete_analysis", END)
    builder.add_edge("mark_invalid", END)

    return builder.compile()


job_analysis_graph = build_job_analysis_graph()
