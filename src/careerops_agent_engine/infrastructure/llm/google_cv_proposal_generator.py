"""Gemini implementation of grounded CV proposal generation."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from careerops_agent_engine.agents.prompts.cv_proposal import (
    CV_PROPOSAL_PROMPT,
    PROMPT_VERSION,
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


class GoogleCVProposalGenerator:
    """Generate grounded CV proposals using Gemini."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        """Initialise the structured-output model."""

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=max_retries,
            thinking_level="minimal",
        )

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

        evidence_context = [
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

        messages = CV_PROPOSAL_PROMPT.format_messages(
            requirement=requirement.model_dump_json(indent=2),
            evidence_match=evidence_match.model_dump_json(indent=2),
            approved_evidence=json.dumps(
                [context.model_dump(mode="json") for context in evidence_context],
                indent=2,
            ),
        )

        raw_result: Any = self._structured_model.invoke(
            messages,
            config={
                "run_name": "generate_cv_proposal",
                "tags": [
                    "careerops",
                    "cv-proposal",
                    "structured-output",
                ],
                "metadata": {
                    "job_id": job_id,
                    "requirement_id": (requirement.requirement_id),
                    "prompt_version": PROMPT_VERSION,
                    "model_name": self._model_name,
                },
            },
        )

        generated = GeneratedCVProposalContent.model_validate(raw_result)

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
