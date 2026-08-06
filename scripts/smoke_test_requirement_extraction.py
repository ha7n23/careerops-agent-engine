"""Run one real structured job-requirement extraction."""

from langchain_core.tracers.langchain import wait_for_all_tracers

from careerops_agent_engine.infrastructure.llm.factory import (
    create_requirement_extractor,
)

SYNTHETIC_JOB_DESCRIPTION = """
Junior AI Engineer

We are looking for a Junior AI Engineer to build production-quality
Generative AI applications.

Essential requirements:
- Strong Python software-engineering skills.
- Experience building stateful workflows with LangGraph.
- Experience developing APIs with FastAPI.
- Understanding of Docker-based containerisation.

Desirable requirements:
- Experience using LangSmith for tracing and evaluation.
- Familiarity with AWS cloud deployment.
- Exposure to Kubernetes is beneficial but not required.

The successful candidate will write tested, maintainable code and work
with engineers to evaluate and monitor Large Language Model applications.
""".strip()


def main() -> None:
    """Extract and print validated requirements."""

    extractor = create_requirement_extractor()

    extraction = extractor.extract(
        SYNTHETIC_JOB_DESCRIPTION,
        job_id="JOB-SMOKE-001",
    )

    print(extraction.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        wait_for_all_tracers()
