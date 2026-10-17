# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS base

COPY --from=ghcr.io/astral-sh/uv:0.9.26@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    HOME=/tmp/poc-home \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/poc /tmp/poc-home /tmp/uv-cache \
    && chmod 1777 /tmp/poc-home /tmp/uv-cache \
    && git config --system --add safe.directory /workspace

WORKDIR /workspace
COPY Global/pyproject.toml Global/uv.lock Global/.python-version ./Global/

# Refuse stale dependencies after a branch switch or a lock-file update.
RUN sha256sum Global/pyproject.toml Global/uv.lock Global/.python-version > /opt/poc/dependencies.sha256

ENTRYPOINT ["poc-container"]
CMD ["python", "Global/scripts/start_workspace.py"]

# Source changes must not invalidate the large dependency installation layers.
FROM base AS cpu-dependencies
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project Global --locked --extra ecg --group dev --no-install-project

FROM base AS gpu-dependencies
# The CUDA wheel and its runtime libraries are selected by the locked gpu extra.
RUN --mount=type=cache,target=/tmp/uv-cache \
    test "$(uname -m)" = x86_64 \
    && uv sync --project Global --locked --extra ecg --extra gpu --group dev --no-install-project

FROM cpu-dependencies AS cpu
COPY Global/src/ ./Global/src/
COPY Global/scripts/ ./Global/scripts/
COPY --chmod=755 Global/scripts/container_entrypoint.py /usr/local/bin/poc-container
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project Global --locked --extra ecg --group dev

FROM gpu-dependencies AS gpu
COPY Global/src/ ./Global/src/
COPY Global/scripts/ ./Global/scripts/
COPY --chmod=755 Global/scripts/container_entrypoint.py /usr/local/bin/poc-container
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --project Global --locked --extra ecg --extra gpu --group dev
