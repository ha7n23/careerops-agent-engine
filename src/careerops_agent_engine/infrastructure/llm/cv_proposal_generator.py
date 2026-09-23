"""LangChain implementation of grounded CV proposal generation."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.agents.prompts.cv_proposal import (
    CV_PROPOSAL_PROMPT,
    CV_PROPOSAL_REGENERATION_PROMPT,
    PROMPT_VERSION,
    REGENERATION_PROMPT_VERSION,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.llm.schemas import (
    GeneratedCVProposalContent,
    ProposalEvidenceContext,
)
from careerops_agent_engine.infrastructure.observability.langsmith import (
    build_langsmith_run_config,
)


class LangChainCVProposalGenerator:
    """Generate grounded CV proposals with a configured chat model."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        model_name: str,
    ) -> None:
        """Initialise provider-neutral structured output."""

        self._structured_model = model.with_structured_output(
            schema=GeneratedCVProposalContent.model_json_schema(),
            method="json_schema",
        )
        self._model_name = model_name

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Generate one proposal from validated direct evidence."""

        evidence_context = build_evidence_context(approved_evidence)

        messages = CV_PROPOSAL_PROMPT.format_messages(
            requirement=requirement.model_dump_json(indent=2),
            evidence_match=evidence_match.model_dump_json(indent=2),
            approved_evidence=json.dumps(
                [context.model_dump(mode="json") for context in evidence_context],
                indent=2,
            ),
        )

        generated = self._invoke(
            messages=messages,
            run_name="generate_cv_proposal",
            requirement=requirement,
            prompt_version=PROMPT_VERSION,
        )

        return build_domain_proposal(
            proposal_id=proposal_id,
            requirement=requirement,
            generated=generated,
        )

    def regenerate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
        previous_proposal: CVChangeProposal,
        reviewer_feedback: str,
    ) -> CVChangeProposal:
        """Regenerate a proposal using grounded human feedback."""

        evidence_context = build_evidence_context(approved_evidence)

        messages = CV_PROPOSAL_REGENERATION_PROMPT.format_messages(
            requirement=requirement.model_dump_json(indent=2),
            evidence_match=(evidence_match.model_dump_json(indent=2)),
            approved_evidence=json.dumps(
                [context.model_dump(mode="json") for context in evidence_context],
                indent=2,
            ),
            previous_proposal=(previous_proposal.model_dump_json(indent=2)),
            reviewer_feedback=reviewer_feedback,
        )

        generated = self._invoke(
            messages=messages,
            run_name="regenerate_cv_proposal",
            requirement=requirement,
            prompt_version=(REGENERATION_PROMPT_VERSION),
        )

        return build_domain_proposal(
            proposal_id=proposal_id,
            requirement=requirement,
            generated=generated,
        )

    def _invoke(
        self,
        *,
        messages: list[Any],
        run_name: str,
        requirement: JobRequirement,
        prompt_version: str,
    ) -> GeneratedCVProposalContent:
        """Invoke the configured model and validate structured content."""

        raw_result: Any = self._structured_model.invoke(
            messages,
            config=build_langsmith_run_config(
                run_name=run_name,
                tags=[
                    "cv-proposal",
                    "structured-output",
                    "llm",
                ],
                metadata={
                    "component": ("cv_proposal_generator"),
                    "requirement_id": (requirement.requirement_id),
                    "prompt_version": (prompt_version),
                    "ls_model_name": (self._model_name),
                },
            ),
        )

        return GeneratedCVProposalContent.model_validate(raw_result)


def build_evidence_context(
    approved_evidence: Sequence[CareerEvidence],
) -> list[ProposalEvidenceContext]:
    """Build the minimal approved context exposed to the model."""

    return [
        ProposalEvidenceContext(
            evidence_id=evidence.evidence_id,
            category=evidence.category,
            title=evidence.title,
            technologies=list(evidence.technologies),
            capabilities=list(evidence.capabilities),
            approved_claims=list(evidence.approved_claims),
        )
        for evidence in approved_evidence
    ]


def build_domain_proposal(
    *,
    proposal_id: str,
    requirement: JobRequirement,
    generated: GeneratedCVProposalContent,
) -> CVChangeProposal:
    """Attach trusted deterministic metadata to model output."""

    return CVChangeProposal(
        proposal_id=proposal_id,
        section=generated.section,
        target_entry_id=None,
        current_text=None,
        proposed_text=generated.proposed_text,
        requirement_ids=[requirement.requirement_id],
        supporting_evidence_ids=(generated.supporting_evidence_ids),
        confidence_score=generated.confidence_score,
        warnings=generated.warnings,
    )
