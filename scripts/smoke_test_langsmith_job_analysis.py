"""Live proof of CareerOps LangSmith tracing and privacy boundaries."""

import json
import os
import time
from collections.abc import Iterable
from uuid import uuid4

from langchain_core.tracers.langchain import (
    wait_for_all_tracers,
)
from langsmith import Client
from langsmith.schemas import Run
from sqlalchemy import delete

from careerops_agent_engine.api.dependencies import (
    get_database_session_factory,
    get_job_analysis_service,
)
from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceSourceType,
    ReviewAction,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)
from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)
from careerops_agent_engine.infrastructure.database.models.evidence import (
    CareerEvidenceRecord,
)
from careerops_agent_engine.infrastructure.database.models.job_analysis import (
    CVReviewHistoryRecord,
    JobAnalysisRunRecord,
)

EXPECTED_ROOT_RUN_NAMES = {
    "careerops_job_analysis_start",
    "careerops_job_analysis_review_resume",
}

EXPECTED_CHILD_RUN_NAMES = {
    "extract_job_requirements",
    "discover_requirement_evidence",
    "generate_cv_proposal",
    "verify_cv_claims",
}

TRACE_FETCH_ATTEMPTS = 10
TRACE_FETCH_DELAY_SECONDS = 1.0


def main() -> None:
    """Run and verify one synthetic traced CareerOps workflow."""

    validate_langsmith_environment()

    suffix = uuid4().hex[:8].upper()

    user_id = f"USER-LS-SMOKE-{suffix}"

    job_id = f"JOB-LS-SMOKE-{suffix}"

    evidence_id = f"EVD-LS-SMOKE-{suffix}"

    source_id = f"SRC-LS-SMOKE-{suffix}"

    private_marker = f"TRACE-PRIVATE-JD-{suffix}"

    job_description = (
        "Junior AI Engineer.\n\n"
        "Essential requirement:\n"
        "- Build tested FastAPI services for AI applications.\n\n"
        f"Internal synthetic reference: {private_marker}"
    )

    session_factory = get_database_session_factory()

    thread_id: str | None = None

    seed_synthetic_evidence(
        user_id=user_id,
        evidence_id=evidence_id,
        source_id=source_id,
    )

    try:
        service = get_job_analysis_service()

        print("Starting synthetic traced job analysis...")

        started = service.analyse(
            job_id=job_id,
            user_id=user_id,
            job_description=job_description,
        )

        thread_id = started.thread_id

        print(f"LangGraph thread_id: {thread_id}")

        if not started.awaiting_review:
            raise RuntimeError("Synthetic workflow did not reach human review.")

        reviewable_ids = list(
            started.state.get(
                "reviewable_proposal_ids",
                [],
            )
        )

        if not reviewable_ids:
            raise RuntimeError(
                "Synthetic workflow produced no reviewable CV proposals."
            )

        print(f"Reviewable proposal count: {len(reviewable_ids)}")

        decision = CVReviewDecision(
            action=ReviewAction.APPROVE,
            approved_proposal_ids=(reviewable_ids),
        )

        resumed = service.resume_review(
            thread_id=thread_id,
            user_id=user_id,
            decision=decision,
        )

        if resumed.awaiting_review:
            raise RuntimeError("Approved workflow unexpectedly remained paused.")

        if resumed.state.get("status") != "completed":
            raise RuntimeError("Synthetic workflow did not complete.")

        print("Human-review resume completed.")

        # LangChain submits traces in a background thread.
        # Ensure they have been sent before querying LangSmith.
        wait_for_all_tracers()

        client = Client()

        project_name = require_environment("LANGSMITH_PROJECT")

        root_runs = fetch_root_thread_runs(
            client=client,
            project_name=project_name,
            thread_id=thread_id,
        )

        all_runs = fetch_all_thread_runs(
            client=client,
            project_name=project_name,
            thread_id=thread_id,
        )

        verify_root_trace_contract(
            root_runs=root_runs,
            thread_id=thread_id,
        )

        verify_child_trace_contract(
            runs=all_runs,
        )

        verify_trace_privacy(
            runs=all_runs,
            sensitive_values={
                user_id,
                job_id,
                private_marker,
            },
        )

        print()
        print("=== CareerOps LangSmith live trace proof ===")
        print(f"project: {project_name}")
        print(f"thread_id: {thread_id}")
        print(f"root traces: {len(root_runs)}")
        print(f"total runs: {len(all_runs)}")
        print("start trace: found")
        print("review-resume trace: found")
        print("requirement extraction: traced")
        print("evidence agent: traced")
        print("CV proposal generation: traced")
        print("claim verification: traced")
        print("unsafe synthetic identifiers: absent")
        print("LANGSMITH TRACE + PRIVACY PROOF PASSED")

    finally:
        # LangSmith traces intentionally remain in the tracing
        # project as portfolio/debug evidence. Local synthetic
        # database and checkpoint state are removed.
        if thread_id is not None:
            with open_postgres_checkpointer() as checkpointer:
                checkpointer.delete_thread(thread_id)

        with session_factory.begin() as session:
            if thread_id is not None:
                session.execute(
                    delete(CVReviewHistoryRecord).where(
                        CVReviewHistoryRecord.thread_id == thread_id
                    )
                )

                session.execute(
                    delete(JobAnalysisRunRecord).where(
                        JobAnalysisRunRecord.thread_id == thread_id
                    )
                )

            session.execute(
                delete(CareerEvidenceRecord).where(
                    CareerEvidenceRecord.user_id == user_id,
                    CareerEvidenceRecord.evidence_id == evidence_id,
                )
            )


def validate_langsmith_environment() -> None:
    """Require the intended privacy-first live tracing configuration."""

    if require_environment("LANGSMITH_TRACING").casefold() != "true":
        raise RuntimeError("LANGSMITH_TRACING must be true for this live proof.")

    require_environment("LANGSMITH_API_KEY")

    require_environment("LANGSMITH_PROJECT")

    if require_environment("LANGSMITH_HIDE_INPUTS").casefold() != "true":
        raise RuntimeError("LANGSMITH_HIDE_INPUTS must remain true.")

    if require_environment("LANGSMITH_HIDE_OUTPUTS").casefold() != "true":
        raise RuntimeError("LANGSMITH_HIDE_OUTPUTS must remain true.")


def require_environment(
    name: str,
) -> str:
    """Return one required nonblank environment variable."""

    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(f"{name} is required.")

    return value


def seed_synthetic_evidence(
    *,
    user_id: str,
    evidence_id: str,
    source_id: str,
) -> None:
    """Insert one approved synthetic evidence record."""

    session_factory = get_database_session_factory()

    with session_factory.begin() as session:
        session.add(
            CareerEvidenceRecord(
                user_id=user_id,
                evidence_id=evidence_id,
                category=(EvidenceCategory.PROJECT.value),
                title=("Synthetic CareerOps AI Engineering Project"),
                verification_status=(VerificationStatus.APPROVED.value),
                technologies=[
                    "Python",
                    "FastAPI",
                    "LangGraph",
                    "LangSmith",
                    "Docker",
                ],
                capabilities=[
                    ("Stateful AI agent workflow development"),
                    ("Human-in-the-loop workflow design"),
                    ("AI API development with FastAPI"),
                    ("LLM tracing and observability"),
                    ("Python application containerisation"),
                ],
                approved_claims=[
                    (
                        "Built stateful LangGraph "
                        "AI workflows with "
                        "human-in-the-loop review."
                    ),
                    ("Developed tested FastAPI services for AI applications."),
                    ("Integrated LangSmith tracing for AI observability."),
                    ("Containerised Python API applications using Docker."),
                ],
                source_references=[
                    {
                        "source_type": (EvidenceSourceType.MANUAL_ENTRY.value),
                        "source_id": source_id,
                        "page_number": None,
                        "source_excerpt": None,
                    }
                ],
            )
        )


def build_thread_filter(
    thread_id: str,
) -> str:
    """Build the LangSmith filter for one CareerOps thread."""

    return f'and(eq(metadata_key, "thread_id"), eq(metadata_value, "{thread_id}"))'


def fetch_root_thread_runs(
    *,
    client: Client,
    project_name: str,
    thread_id: str,
) -> list[Run]:
    """Wait for both root traces to become queryable."""

    for _ in range(TRACE_FETCH_ATTEMPTS):
        runs = list(
            client.list_runs(
                project_name=project_name,
                filter=(build_thread_filter(thread_id)),
                is_root=True,
            )
        )

        names = {run.name for run in runs}

        if names >= EXPECTED_ROOT_RUN_NAMES:
            return runs

        time.sleep(TRACE_FETCH_DELAY_SECONDS)

    raise RuntimeError("LangSmith did not expose both CareerOps root traces in time.")


def fetch_all_thread_runs(
    *,
    client: Client,
    project_name: str,
    thread_id: str,
) -> list[Run]:
    """Fetch every run carrying the shared thread metadata."""

    runs = list(
        client.list_runs(
            project_name=project_name,
            filter=(build_thread_filter(thread_id)),
        )
    )

    if not runs:
        raise RuntimeError("LangSmith returned no runs for the CareerOps thread.")

    return runs


def verify_root_trace_contract(
    *,
    root_runs: list[Run],
    thread_id: str,
) -> None:
    """Require the start and resume traces under one thread."""

    runs_by_name = {run.name: run for run in root_runs}

    missing = EXPECTED_ROOT_RUN_NAMES - set(runs_by_name)

    if missing:
        raise RuntimeError(
            "Missing LangSmith root traces: " + ", ".join(sorted(missing))
        )

    for expected_name in EXPECTED_ROOT_RUN_NAMES:
        run = runs_by_name[expected_name]

        metadata = run_metadata(run)

        if metadata.get("thread_id") != thread_id:
            raise RuntimeError(f"{expected_name} has the wrong LangSmith thread_id.")

        if metadata.get("workflow") != "job-analysis":
            raise RuntimeError(f"{expected_name} is missing workflow metadata.")

        if metadata.get("workflow_version") != "job-analysis-v1":
            raise RuntimeError(f"{expected_name} is missing workflow-version metadata.")


def verify_child_trace_contract(
    *,
    runs: list[Run],
) -> None:
    """Require the important AI stages to be observable."""

    run_names = {run.name for run in runs}

    missing = EXPECTED_CHILD_RUN_NAMES - run_names

    if missing:
        raise RuntimeError(
            "Missing expected LangSmith child runs: " + ", ".join(sorted(missing))
        )

    metadata_values = [run_metadata(run) for run in runs]

    if not any(metadata.get("prompt_version") for metadata in metadata_values):
        raise RuntimeError("No prompt-version metadata was observed.")

    if not any(metadata.get("ls_model_name") for metadata in metadata_values):
        raise RuntimeError("No model-name metadata was observed.")


def verify_trace_privacy(
    *,
    runs: Iterable[Run],
    sensitive_values: set[str],
) -> None:
    """Ensure synthetic private values did not reach LangSmith."""

    for run in runs:
        serialized = json.dumps(
            {
                "name": run.name,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "extra": run.extra,
                "tags": run.tags,
            },
            default=str,
            sort_keys=True,
        )

        for sensitive_value in sensitive_values:
            if sensitive_value in serialized:
                raise RuntimeError(
                    "Sensitive synthetic value "
                    "was found in LangSmith run "
                    f"{run.name!r}."
                )


def run_metadata(
    run: Run,
) -> dict[str, object]:
    """Return one run's metadata dictionary safely."""

    extra = run.extra or {}

    metadata = extra.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        return {}

    return {str(key): value for key, value in metadata.items()}


if __name__ == "__main__":
    main()
