"""LangChain implementation of strict CV claim verification."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.agents.prompts.cv_claim_verification import (
    BATCH_PROMPT_VERSION,
    CV_CLAIM_VERIFICATION_BATCH_PROMPT,
    CV_CLAIM_VERIFICATION_PROMPT,
    PROMPT_VERSION,
)
from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerificationRequest,
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
    GeneratedClaimVerificationBatch,
    GeneratedClaimVerificationBatchItem,
    ProposalEvidenceContext,
)
from careerops_agent_engine.infrastructure.observability.langsmith import (
    build_langsmith_run_config,
)


class LangChainCVClaimVerifier:
    """Verify generated CV claims with a configured chat model."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        model_name: str,
    ) -> None:
        """Initialise provider-neutral structured output."""

        self._structured_model = model.with_structured_output(
            schema=GeneratedClaimVerification.model_json_schema(),
            method="json_schema",
        )
        self._batch_structured_model = model.with_structured_output(
            schema=GeneratedClaimVerificationBatch.model_json_schema(),
            method="json_schema",
        )
        self._model_name = model_name

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Verify every factual claim in one proposal."""

        evidence_context = build_evidence_context(approved_evidence)

        messages = CV_CLAIM_VERIFICATION_PROMPT.format_messages(
            proposed_text=proposal.proposed_text,
            approved_evidence=json.dumps(
                [context.model_dump(mode="json") for context in evidence_context],
                indent=2,
            ),
        )

        raw_result: Any = self._structured_model.invoke(
            messages,
            config=build_langsmith_run_config(
                run_name="verify_cv_claims",
                tags=[
                    "cv-claim-verification",
                    "structured-output",
                    "llm",
                ],
                metadata={
                    "component": "cv_claim_verifier",
                    "proposal_id": proposal.proposal_id,
                    "prompt_version": PROMPT_VERSION,
                    "ls_model_name": self._model_name,
                },
            ),
        )

        generated = GeneratedClaimVerification.model_validate(raw_result)

        return build_domain_report(
            proposal_id=proposal.proposal_id,
            generated=generated,
        )

    def verify_batch(
        self,
        *,
        requests: Sequence[CVClaimVerificationRequest],
    ) -> list[CVClaimVerificationReport]:
        """Verify multiple proposals through one model invocation."""

        request_list = list(requests)

        if not request_list:
            return []

        requests_by_proposal: dict[
            str,
            CVClaimVerificationRequest,
        ] = {}
        request_context: list[dict[str, Any]] = []

        for request in request_list:
            proposal_id = request.proposal.proposal_id

            if proposal_id in requests_by_proposal:
                raise ValueError(
                    "Batch verification requests require unique proposal identifiers."
                )

            requests_by_proposal[proposal_id] = request

            evidence_context = build_evidence_context(request.approved_evidence)

            request_context.append(
                {
                    "proposal_id": proposal_id,
                    "proposal": request.proposal.model_dump(
                        mode="json",
                    ),
                    "approved_evidence": [
                        context.model_dump(mode="json") for context in evidence_context
                    ],
                }
            )

        messages = CV_CLAIM_VERIFICATION_BATCH_PROMPT.format_messages(
            verification_requests=json.dumps(
                request_context,
                indent=2,
            )
        )

        raw_result: Any = self._batch_structured_model.invoke(
            messages,
            config=build_langsmith_run_config(
                run_name="verify_cv_claim_batch",
                tags=[
                    "cv-claim-verification",
                    "batch",
                    "structured-output",
                    "llm",
                ],
                metadata={
                    "component": "cv_claim_verifier",
                    "prompt_version": BATCH_PROMPT_VERSION,
                    "ls_model_name": self._model_name,
                },
            ),
        )

        generated_batch = GeneratedClaimVerificationBatch.model_validate(raw_result)

        generated_by_proposal: dict[
            str,
            GeneratedClaimVerificationBatchItem,
        ] = {}

        for generated in generated_batch.verifications:
            if generated.proposal_id in generated_by_proposal:
                raise ValueError(
                    "Batch verification response contains duplicate "
                    "proposal identifiers."
                )

            generated_by_proposal[generated.proposal_id] = generated

        if set(generated_by_proposal) != set(requests_by_proposal):
            raise ValueError(
                "Batch verification response must contain exactly "
                "one verification per requested proposal."
            )

        return [
            build_domain_report(
                proposal_id=request.proposal.proposal_id,
                generated=generated_by_proposal[request.proposal.proposal_id],
            )
            for request in request_list
        ]


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


def build_domain_report(
    *,
    proposal_id: str,
    generated: GeneratedClaimVerification,
) -> CVClaimVerificationReport:
    """Attach trusted proposal identity and summary fields."""

    claims = [
        ClaimAssessment(
            claim_text=claim.claim_text,
            supported=claim.supported,
            supporting_evidence_ids=(claim.supporting_evidence_ids),
            explanation=claim.explanation,
        )
        for claim in generated.claims
    ]

    unsupported_claims = [claim.claim_text for claim in claims if not claim.supported]
    fully_supported = generated.coverage_complete and not unsupported_claims

    return CVClaimVerificationReport(
        proposal_id=proposal_id,
        claims=claims,
        coverage_complete=generated.coverage_complete,
        coverage_notes=generated.coverage_notes,
        fully_supported=fully_supported,
        unsupported_claims=unsupported_claims,
    )
