# patpy-mcp

[![BioContextAI - Registry](https://img.shields.io/badge/Registry-package?style=flat&label=BioContextAI&labelColor=%23fff&color=%233555a1&link=https://biocontext.ai/registry)](https://biocontext.ai/registry)
[![PyPI](https://img.shields.io/pypi/v/patpy-mcp?label=PyPI)](https://pypi.org/project/patpy-mcp/)

`patpy-mcp` is an MCP (Model Context Protocol) server that lets any
MCP-capable LLM agent discover and download single-cell datasets from
public registries — currently CellxGene Discover. It is part of the
[BioContextAI Registry](https://biocontext.ai/registry) and is built
from the
[`biocontext-ai/mcp-server-cookiecutter`](https://github.com/biocontext-ai/mcp-server-cookiecutter)
template, so its layout (one tool per file under `tools/`, click CLI in
`main.py`, module-level `FastMCP` in `mcp.py`) matches every other
BioContextAI server.

`patpy-mcp` lives in the [`patpy`](https://github.com/lueckenlab/patpy)
monorepo as a self-contained sub-project under [`mcp/`](.) and is
released to PyPI independently of the parent `patpy` package.

## Quick start

You need Python ≥ 3.11. If you don't have Python yet, install [`uv`](https://github.com/astral-sh/uv) — it bootstraps Python and runs `patpy-mcp` in one step.

There are four equivalent ways to install / run `patpy-mcp`, mirroring the four patterns from the BioContextAI [`mcp-server-cookiecutter`](https://github.com/biocontext-ai/mcp-server-cookiecutter):

### 1. Run the latest published release on demand (recommended)

```bash
uvx patpy-mcp
```

### 2. Add it to a client that supports the `mcp.json` standard

Cursor, Claude Desktop, Continue.dev, mcp-cli, Goose, etc. all read this exact JSON shape — copy it verbatim from [`mcp.json`](mcp.json) and drop it into your client config.

**From PyPI** (after release):

```json
{
  "mcpServers": {
    "lueckenlab/patpy-mcp": {
      "command": "uvx",
      "args": ["patpy-mcp"]
    }
  }
}
```

**From the GitHub `main` branch** (before any release tag):

```json
{
  "mcpServers": {
    "lueckenlab/patpy-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/lueckenlab/patpy.git@main#subdirectory=mcp",
        "patpy-mcp"
      ]
    }
  }
}
```

**From a local checkout** (development):

```json
{
  "mcpServers": {
    "lueckenlab/patpy-mcp": {
      "command": "uvx",
      "args": ["--refresh", "--from", "/abs/path/to/patpy/mcp", "patpy-mcp"]
    }
  }
}
```

### 3. Install with `pip`

```bash
pip install --user patpy-mcp
patpy-mcp                       # stdio transport (default)
patpy-mcp --transport http      # HTTP transport for remote clients
patpy-mcp --version
```

### 4. Run via Docker

Build context is the repo root (so the shared top-level `LICENSE` is present):

```bash
docker build -t patpy-mcp -f mcp/Dockerfile .
docker run --rm -i patpy-mcp
```

## What it exposes

| Tool                              | Purpose                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| `list_sources`                    | List dataset sources enabled in this server build.                                   |
| `describe_source`                 | Description and capabilities for one source.                                         |
| `cellxgene_search_datasets`       | Search CellxGene Discover by disease, tissue, organism, assay, or free-text title.   |
| `cellxgene_get_dataset`           | Full metadata for a dataset, including downloadable assets.                          |
| `cellxgene_list_collections`      | List CellxGene collections (publications), optionally filtered by free text.         |
| `cellxgene_get_collection`        | Full metadata for a collection, including its datasets.                              |
| `cellxgene_list_disease_terms`    | Distinct disease ontology terms present in CellxGene (label + ontology ID).          |
| `cellxgene_list_tissue_terms`     | Distinct tissue ontology terms present in CellxGene.                                 |
| `cellxgene_download_dataset`      | Stream-download a dataset asset to the local cache, returning path / size / SHA-256. |

For agent configuration snippets (Claude Desktop, Cursor, mcp-cli +
Ollama, …) and a sample workflow, see
[`docs/mcp.md`](../docs/mcp.md) at the repo root.

## How it complements other BioContextAI servers

`patpy-mcp` deliberately stops at *dataset discovery and download* and
defers neighbouring concerns to existing community servers:

- [`MaxMLang/cxg-census-mcp`](https://github.com/MaxMLang/cxg-census-mcp)
  for Census slice queries (TileDB-SOMA).
- [`biocontext-ai/anndata-mcp`](https://github.com/biocontext-ai/anndata-mcp)
  for AnnData inspection. Files downloaded here can be passed straight
  to `anndata-mcp` by absolute path because both servers share the
  `~/.cache/patpy-mcp/` layout.

## Layout

```
mcp/
├── pyproject.toml          # standalone PyPI package (build = hatchling)
├── README.md               # this file
├── CITATION.cff
├── meta.yaml               # BioContextAI Registry entry (Schema.org metadata)
├── mcp.json                # BioContextAI Registry entry (MCP client config snippet)
├── Dockerfile              # slim deploy image
├── src/patpy_mcp/
│   ├── __init__.py
│   ├── main.py             # click CLI (run_app)
│   ├── mcp.py              # module-level FastMCP instance
│   ├── cache.py            # on-disk cache layout & sidecars
│   ├── sources/            # data-source descriptors + REST clients
│   └── tools/_*.py         # one @mcp.tool function per file
└── tests/
    ├── conftest.py         # isolated_cache autouse fixture
    ├── test_app.py         # CLI + tool registration smoke tests
    ├── test_cellxgene_discover.py
    └── test_registry_meta.py
```

## Submitting / updating the BioContextAI Registry entry

A registry entry is a directory under
[`biocontext-ai/registry/servers/<owner>-<name>/`](https://github.com/biocontext-ai/registry/tree/main/servers)
containing **two files**:

- `meta.yaml` — Schema.org / JSON-LD metadata (validated against the
  upstream JSON schema).
- `mcp.json` — a small MCP client-config snippet that tells any
  MCP-compatible LLM agent how to launch this server in one line, e.g.
  `uvx patpy-mcp`.

To submit / update:

1. Validate `meta.yaml` against the registry schema locally:

   ```bash
   pytest mcp/tests/test_registry_meta.py
   ```

2. Fork [`biocontext-ai/registry`](https://github.com/biocontext-ai/registry)
   and copy both files into
   `servers/lueckenlab-patpy-mcp/`:

   ```text
   servers/lueckenlab-patpy-mcp/meta.yaml   # from mcp/meta.yaml
   servers/lueckenlab-patpy-mcp/mcp.json    # from mcp/mcp.json
   ```

3. Open a PR; the upstream `pre-commit` hook re-validates the entry.

## Releasing to PyPI

Push a tag of the form `patpy-mcp-v0.1.0` to the repo. The
[`release-patpy-mcp.yaml`](../.github/workflows/release-patpy-mcp.yaml)
workflow runs `uv build` inside `mcp/` and uploads the resulting
distribution to PyPI via trusted publishing — `patpy` and `patpy-mcp`
release independently from the same monorepo.
