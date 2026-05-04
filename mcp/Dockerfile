# syntax=docker/dockerfile:1.7
# Slim image that installs patpy with MCP extras and runs the stdio server.
# Build from the repository root: `docker build -t patpy-mcp -f mcp/Dockerfile .`
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATPY_MCP_CACHE=/data/cache

RUN useradd --create-home --shell /bin/bash patpy

WORKDIR /app

# Copy only what is needed to install the package; the source is small enough
# that we don't bother with a separate dependency-only layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --upgrade pip \
    && pip install '.[mcp]'

USER patpy
RUN mkdir -p "${PATPY_MCP_CACHE}"
VOLUME ["/data/cache"]

# patpy-mcp speaks MCP over stdio, so we keep STDIN open and STDOUT clean.
ENTRYPOINT ["patpy-mcp"]
