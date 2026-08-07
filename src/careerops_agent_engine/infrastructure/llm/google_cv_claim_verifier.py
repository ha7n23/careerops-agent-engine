"""Gemini implementation of strict CV claim verification."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from careerops_agent_engine.agents.prompts.cv_claim_verification import (
    CV_CLAIM_VERIFICATION_PROMPT,
    PROMPT_VERSION,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)
from careerops_agent_engine.domain.models.verification import (
    ClaimAssessment,
    CVClaimVerificationReport,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    GeneratedClaimVerification,
    ProposalEvidenceContext,
)


class GoogleCVClaimVerifier:
    """Verify generated CV claims against approved evidence."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        """Initialise the structured verification model."""

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=max_retries,
            thinking_level="minimal",
        )

        self._structured_model = model.with_structured_output(
            schema=GeneratedClaimVerification.model_json_schema(),
            method="json_schema",
        )

        self._model_name = model_name

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Verify every factual claim in the proposal."""

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

        messages = CV_CLAIM_VERIFICATION_PROMPT.format_messages(
            proposed_text=proposal.proposed_text,
            approved_evidence=json.dumps(
                [context.model_dump(mode="json") for context in evidence_context],
                indent=2,
            ),
        )

        raw_result: Any = self._structured_model.invoke(
            messages,
            config={
                "run_name": "verify_cv_claims",
                "tags": [
                    "careerops",
                    "cv-claim-verification",
                    "structured-output",
                ],
                "metadata": {
                    "proposal_id": proposal.proposal_id,
                    "prompt_version": PROMPT_VERSION,
                    "model_name": self._model_name,
                },
            },
        )

        generated = GeneratedClaimVerification.model_validate(raw_result)

        claims = [
            ClaimAssessment(
                claim_text=claim.claim_text,
                supported=claim.supported,
                supporting_evidence_ids=(claim.supporting_evidence_ids),
                explanation=claim.explanation,
            )
            for claim in generated.claims
        ]

        unsupported_claims = [
            claim.claim_text for claim in claims if not claim.supported
        ]

        fully_supported = generated.coverage_complete and not unsupported_claims

        return CVClaimVerificationReport(
            proposal_id=proposal.proposal_id,
            claims=claims,
            coverage_complete=generated.coverage_complete,
            coverage_notes=generated.coverage_notes,
            fully_supported=fully_supported,
            unsupported_claims=unsupported_claims,
        )
