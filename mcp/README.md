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

```bash
# Recommended: run the latest release on demand without polluting your env
uvx patpy-mcp

# Or install from PyPI:
pip install patpy-mcp
patpy-mcp                       # stdio MCP server (default)
patpy-mcp --transport http      # HTTP transport for remote clients
patpy-mcp --version
```

Or via Docker, from the repo root (so the shared `LICENSE` is in the context):

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
├── meta.yaml               # BioContextAI Registry entry
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

1. Validate `meta.yaml` against the registry schema:

   ```bash
   pytest mcp/tests/test_registry_meta.py
   ```

2. Fork [`biocontext-ai/registry`](https://github.com/biocontext-ai/registry)
   and copy this `mcp/meta.yaml` to
   `servers/lueckenlab-patpy/meta.yaml`.
3. Open a PR; the upstream `pre-commit` hook re-validates the file.

## Releasing to PyPI

Push a tag of the form `patpy-mcp-v0.1.0` to the repo. The
[`release-patpy-mcp.yaml`](../.github/workflows/release-patpy-mcp.yaml)
workflow runs `uv build` inside `mcp/` and uploads the resulting
distribution to PyPI via trusted publishing — `patpy` and `patpy-mcp`
release independently from the same monorepo.
