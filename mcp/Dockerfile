# syntax=docker/dockerfile:1.7
# Slim image that installs patpy-mcp from this monorepo and runs the stdio
# server. Build from the repository root so that the shared `LICENSE` file
# (referenced from `mcp/pyproject.toml`) is part of the build context:
#
#     docker build -t patpy-mcp -f mcp/Dockerfile .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATPY_MCP_CACHE=/data/cache

RUN useradd --create-home --shell /bin/bash patpy

WORKDIR /app

# Bring in the LICENSE first so it lives at /app/LICENSE -- mcp/pyproject.toml
# references it as `../LICENSE` relative to /app/mcp/.
COPY LICENSE ./LICENSE

# Then copy the patpy-mcp sub-project itself.
COPY mcp/ ./mcp/

RUN pip install --upgrade pip \
    && pip install ./mcp

USER patpy
RUN mkdir -p "${PATPY_MCP_CACHE}"
VOLUME ["/data/cache"]

# patpy-mcp speaks MCP over stdio by default, so we keep STDIN open and
# STDOUT clean. Override with `--transport http` for network use.
ENTRYPOINT ["patpy-mcp"]
