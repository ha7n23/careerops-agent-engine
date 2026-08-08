"""Construction of the CareerOps job-analysis graph."""

from typing import Literal

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from careerops_agent_engine.agents.nodes.job_analysis import (
    apply_human_edits,
    calculate_fit,
    complete_analysis,
    create_discover_evidence_node,
    create_extract_requirements_node,
    create_generate_cv_proposals_node,
    create_regenerate_cv_proposals_node,
    create_reverify_human_edits_node,
    create_verify_cv_proposals_node,
    create_verify_regenerated_proposals_node,
    finalize_human_review,
    mark_invalid,
    request_edit_rework,
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
from careerops_agent_engine.domain.enums import ReviewAction
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
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
    """Require human review only for verified proposals."""

    if state.get("reviewable_proposal_ids"):
        return "review"

    return "complete"


def route_after_human_review(
    state: JobAnalysisState,
) -> Literal[
    "edit",
    "regenerate",
    "finalize",
]:
    """Route the current human decision."""

    decision_payload = state.get("review_decision")

    if decision_payload is None:
        raise ValueError("Human-review routing requires a decision.")

    decision = CVReviewDecision.model_validate(decision_payload)

    if decision.action is ReviewAction.EDIT:
        return "edit"

    if decision.action is ReviewAction.REGENERATE:
        return "regenerate"

    return "finalize"


def route_after_edit_verification(
    state: JobAnalysisState,
) -> Literal["passed", "failed"]:
    """Determine whether edited wording may be finalized."""

    if state.get("edit_verification_failed_ids"):
        return "failed"

    return "passed"


def route_after_regeneration_verification(
    state: JobAnalysisState,
) -> Literal["passed", "failed"]:
    """Determine whether regenerated wording passed verification."""

    if state.get("regeneration_verification_failed_ids"):
        return "failed"

    return "passed"


def build_job_analysis_graph(
    requirement_extractor: RequirementExtractor,
    evidence_discovery_runner: EvidenceDiscoveryRunner,
    cv_proposal_service: CVProposalGenerationService,
    cv_claim_verification_service: (CVClaimVerificationService),
    checkpointer: (BaseCheckpointSaver[str] | None) = None,
) -> CompiledStateGraph[
    JobAnalysisState,
    None,
    JobAnalysisState,
    JobAnalysisState,
]:
    """Compile the durable CareerOps job-analysis workflow."""

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

    regenerate_cv_proposals_node = RunnableLambda(
        create_regenerate_cv_proposals_node(cv_proposal_service)
    )

    verify_regenerated_proposals_node = RunnableLambda(
        create_verify_regenerated_proposals_node(cv_claim_verification_service)
    )

    reverify_human_edits_node = RunnableLambda(
        create_reverify_human_edits_node(cv_claim_verification_service)
    )

    # Initial analysis.
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

    # Initial CV proposal lifecycle.
    builder.add_node(
        "generate_cv_proposals",
        generate_cv_proposals_node,
    )
    builder.add_node(
        "verify_cv_proposals",
        verify_cv_proposals_node,
    )

    # Human review.
    builder.add_node(
        "request_human_review",
        request_human_review,
    )

    # Human editing.
    builder.add_node(
        "apply_human_edits",
        apply_human_edits,
    )
    builder.add_node(
        "reverify_human_edits",
        reverify_human_edits_node,
    )

    # Feedback-aware regeneration.
    builder.add_node(
        "regenerate_cv_proposals",
        regenerate_cv_proposals_node,
    )
    builder.add_node(
        "verify_regenerated_proposals",
        verify_regenerated_proposals_node,
    )

    # Rework and finalization.
    builder.add_node(
        "request_edit_rework",
        request_edit_rework,
    )
    builder.add_node(
        "finalize_human_review",
        finalize_human_review,
    )

    # Terminal nodes.
    builder.add_node(
        "complete_analysis",
        complete_analysis,
    )
    builder.add_node(
        "mark_invalid",
        mark_invalid,
    )

    # Entry.
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

    # Main analysis pipeline.
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

    # Human decision:
    # approve/reject -> finalize
    # edit -> deterministic human edit path
    # regenerate -> feedback-aware generation path
    builder.add_conditional_edges(
        "request_human_review",
        route_after_human_review,
        {
            "edit": "apply_human_edits",
            "regenerate": ("regenerate_cv_proposals"),
            "finalize": "finalize_human_review",
        },
    )

    # Human-edited text must be reverified.
    builder.add_edge(
        "apply_human_edits",
        "reverify_human_edits",
    )

    builder.add_conditional_edges(
        "reverify_human_edits",
        route_after_edit_verification,
        {
            "passed": "finalize_human_review",
            "failed": "request_edit_rework",
        },
    )

    # Regenerated text must also be verified.
    builder.add_edge(
        "regenerate_cv_proposals",
        "verify_regenerated_proposals",
    )

    # IMPORTANT:
    # successful regeneration returns to human review.
    # It never auto-approves itself.
    builder.add_conditional_edges(
        "verify_regenerated_proposals",
        route_after_regeneration_verification,
        {
            "passed": "request_human_review",
            "failed": "request_edit_rework",
        },
    )

    # Rework after an unsafe edit or regeneration.
    builder.add_conditional_edges(
        "request_edit_rework",
        route_after_human_review,
        {
            "edit": "apply_human_edits",
            "regenerate": ("regenerate_cv_proposals"),
            "finalize": "finalize_human_review",
        },
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
