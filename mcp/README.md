# patpy-mcp

[![BioContextAI - Registry](https://img.shields.io/badge/Registry-package?style=flat&label=BioContextAI&labelColor=%23fff&color=%233555a1&link=https://biocontext.ai/registry)](https://biocontext.ai/registry)

MCP server for sample-level single-cell dataset discovery, shipping inside [`patpy`](https://github.com/lueckenlab/patpy).

This directory contains the artifacts that make `patpy-mcp` a first-class entry in the [BioContextAI Registry](https://biocontext.ai/registry):

- `meta.yaml` — Schema.org `SoftwareApplication` record. The registry submission is a copy of this file at `biocontext-ai/registry/servers/lueckenlab-patpy/meta.yaml`.
- `Dockerfile` — slim Python image that installs `patpy[mcp]` and runs `patpy-mcp` over stdio.
- `.dockerignore` — keeps the build context small.

The server itself lives under [`src/patpy/mcp/`](../src/patpy/mcp/). User-facing documentation is in [`docs/mcp.md`](../docs/mcp.md).

## Quick start

```bash
pip install 'patpy[mcp]'
patpy-mcp           # launches the stdio MCP server
```

Or via Docker, from the repo root:

```bash
docker build -t patpy-mcp -f mcp/Dockerfile .
docker run --rm -i patpy-mcp
```

## Submitting / updating the registry entry

1. Validate `meta.yaml` against the registry schema:

   ```bash
   pytest tests/test_mcp_registry_meta.py
   ```

2. Fork [`biocontext-ai/registry`](https://github.com/biocontext-ai/registry) and copy this `meta.yaml` to `servers/lueckenlab-patpy/meta.yaml`.
3. Open a PR; the upstream `pre-commit` hook re-validates the file.
