# syntax=docker/dockerfile:1

FROM python:3.12.13-slim-trixie

# Keep Python container behaviour predictable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# LibreOffice Writer provides the DOCX -> PDF conversion runtime.
# Liberation fonts provide metric-compatible alternatives for common
# Microsoft fonts such as Arial.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-writer \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Use the same uv version as the local project toolchain.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /app

# Install third-party production dependencies first so this layer can
# remain cached when only application source code changes.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# Copy the CareerOps project after dependency installation.
COPY . .

# Install CareerOps itself into the virtual environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-editable

# Runtime-generated documents must be writable, but the application
# itself does not need to run as root.
RUN useradd \
        --create-home \
        --uid 10001 \
        careerops \
    && mkdir -p \
        /app/.careerops_data/documents \
        /app/.careerops_data/artifacts \
    && chown -R \
        careerops:careerops \
        /app/.careerops_data

USER careerops

EXPOSE 8000

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "scripts/check_api_readiness.py"]

CMD ["uvicorn", "careerops_agent_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "25"]