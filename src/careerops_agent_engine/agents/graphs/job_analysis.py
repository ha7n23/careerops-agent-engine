"""Construction of the CareerOps job-analysis graph."""

from typing import Literal

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from careerops_agent_engine.agents.nodes.job_analysis import (
    calculate_fit,
    complete_analysis,
    create_discover_evidence_node,
    create_extract_requirements_node,
    create_generate_cv_proposals_node,
    create_verify_cv_proposals_node,
    finalize_human_review,
    mark_invalid,
    request_human_review,
    validate_job_input,
)
from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
)
from careerops_agent_engine.application.ports.evidence_discovery import (
    EvidenceDiscoveryRunner,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)


def route_after_validation(
    state: JobAnalysisState,
) -> Literal["valid", "invalid"]:
    """Select the valid or invalid execution branch."""

    if state.get("validation_error") is not None:
        return "invalid"

    return "valid"


def route_after_proposal_verification(
    state: JobAnalysisState,
) -> Literal["review", "complete"]:
    """Require human review only when verified proposals exist."""

    if state.get("reviewable_proposal_ids"):
        return "review"

    return "complete"


def build_job_analysis_graph(
    requirement_extractor: RequirementExtractor,
    evidence_discovery_runner: EvidenceDiscoveryRunner,
    cv_proposal_service: CVProposalGenerationService,
    cv_claim_verification_service: CVClaimVerificationService,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[
    JobAnalysisState,
    None,
    JobAnalysisState,
    JobAnalysisState,
]:
    """Compile the CareerOps job-analysis workflow."""

    builder = StateGraph(JobAnalysisState)

    extract_requirements_node = RunnableLambda(
        create_extract_requirements_node(requirement_extractor)
    )

    discover_evidence_node = RunnableLambda(
        create_discover_evidence_node(evidence_discovery_runner)
    )

    generate_cv_proposals_node = RunnableLambda(
        create_generate_cv_proposals_node(cv_proposal_service)
    )

    verify_cv_proposals_node = RunnableLambda(
        create_verify_cv_proposals_node(cv_claim_verification_service)
    )

    builder.add_node(
        "validate_job_input",
        validate_job_input,
    )
    builder.add_node(
        "extract_requirements",
        extract_requirements_node,
    )
    builder.add_node(
        "discover_evidence",
        discover_evidence_node,
    )
    builder.add_node(
        "calculate_fit",
        calculate_fit,
    )
    builder.add_node(
        "generate_cv_proposals",
        generate_cv_proposals_node,
    )
    builder.add_node(
        "verify_cv_proposals",
        verify_cv_proposals_node,
    )
    builder.add_node(
        "request_human_review",
        request_human_review,
    )
    builder.add_node(
        "finalize_human_review",
        finalize_human_review,
    )
    builder.add_node(
        "complete_analysis",
        complete_analysis,
    )
    builder.add_node(
        "mark_invalid",
        mark_invalid,
    )

    builder.add_edge(
        START,
        "validate_job_input",
    )

    builder.add_conditional_edges(
        "validate_job_input",
        route_after_validation,
        {
            "valid": "extract_requirements",
            "invalid": "mark_invalid",
        },
    )

    builder.add_edge(
        "extract_requirements",
        "discover_evidence",
    )
    builder.add_edge(
        "discover_evidence",
        "calculate_fit",
    )
    builder.add_edge(
        "calculate_fit",
        "generate_cv_proposals",
    )
    builder.add_edge(
        "generate_cv_proposals",
        "verify_cv_proposals",
    )

    builder.add_conditional_edges(
        "verify_cv_proposals",
        route_after_proposal_verification,
        {
            "review": "request_human_review",
            "complete": "complete_analysis",
        },
    )

    builder.add_edge(
        "request_human_review",
        "finalize_human_review",
    )
    builder.add_edge(
        "finalize_human_review",
        "complete_analysis",
    )

    builder.add_edge(
        "complete_analysis",
        END,
    )
    builder.add_edge(
        "mark_invalid",
        END,
    )

    return builder.compile(checkpointer=checkpointer)
