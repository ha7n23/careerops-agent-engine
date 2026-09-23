"""LangChain implementation of structured CV evidence extraction."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.agents.prompts.cv_evidence_extraction import (
    CV_EVIDENCE_EXTRACTION_PROMPT,
    PROMPT_VERSION,
)
from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceCandidate,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedCareerEvidenceSet,
)
from careerops_agent_engine.infrastructure.observability.langsmith import (
    build_langsmith_run_config,
)


class LangChainCVEvidenceExtractor:
    """Extract pending evidence candidates with a configured chat model."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        model_name: str,
    ) -> None:
        """Initialise provider-neutral structured output."""

        self._structured_model = model.with_structured_output(
            schema=ExtractedCareerEvidenceSet.model_json_schema(),
            method="json_schema",
        )
        self._model_name = model_name

    def extract(
        self,
        *,
        document: ParsedCVDocument,
    ) -> list[CareerEvidenceCandidate]:
        """Extract candidate evidence from recognised CV sections."""

        section_context = [
            {
                "section": section.section.value,
                "heading": section.heading,
                "order_index": section.order_index,
                "text": section.text,
            }
            for section in document.sections
        ]

        messages = CV_EVIDENCE_EXTRACTION_PROMPT.format_messages(
            parsed_cv_sections=json.dumps(
                section_context,
                indent=2,
            )
        )

        raw_result: Any = self._structured_model.invoke(
            messages,
            config=build_langsmith_run_config(
                run_name="extract_cv_evidence_candidates",
                tags=[
                    "cv-ingestion",
                    "evidence-extraction",
                    "structured-output",
                    "llm",
                ],
                metadata={
                    "component": "cv_evidence_extractor",
                    "document_id": document.document_id,
                    "prompt_version": PROMPT_VERSION,
                    "ls_model_name": self._model_name,
                },
            ),
        )

        extracted = ExtractedCareerEvidenceSet.model_validate(raw_result)

        return [
            CareerEvidenceCandidate(
                category=candidate.category,
                title=candidate.title,
                source_section_order_index=(candidate.source_section_order_index),
                source_excerpt=candidate.source_excerpt,
                technologies=list(candidate.technologies),
                capabilities=list(candidate.capabilities),
                claims=list(candidate.claims),
                warnings=list(candidate.warnings),
            )
            for candidate in extracted.candidates
        ]
