"""CareerOps Agent Engine package."""


def main() -> None:
    """Run the local development server."""

    import uvicorn

    uvicorn.run(
        "careerops_agent_engine.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
