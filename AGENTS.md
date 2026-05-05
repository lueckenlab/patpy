# AGENTS.md

> Instructions for AI coding agents working on this repository.

Read this file first. It tells you where things live, how to install / test, what conventions matter, and where to find more detailed guidance on specific tasks.

## What this repository is

Two related projects share this monorepo:

1. **`patpy`** — a Python package for **sample-level** (donor / patient) representation learning from single-cell data. Operates on `anndata.AnnData` objects where each row is a single cell and each donor contributes many cells. Source: `src/patpy/`. Released to PyPI as [`patpy`](https://pypi.org/project/patpy/).
2. **`patpy-mcp`** — a standalone Model Context Protocol server that exposes CellxGene Discover dataset search and download as MCP tools. Source: `mcp/`. Released to PyPI **independently** as [`patpy-mcp`](https://pypi.org/project/patpy-mcp/) — it does **not** depend on `patpy`. Built with the [BioContextAI cookiecutter](https://github.com/biocontext-ai/mcp-server-cookiecutter) and registered in the [BioContextAI Registry](https://biocontext.ai/registry).

Both ship from this single repo and can be released independently:

- `patpy` — workflows `.github/workflows/test.yaml`, `build.yaml`, `release.yaml`; release tag `v*`.
- `patpy-mcp` — workflows `.github/workflows/test-patpy-mcp.yaml`, `build-patpy-mcp.yaml`, `release-patpy-mcp.yaml`; release tag `patpy-mcp-v*`. All three only run when `mcp/**` changes (path-filtered).

## Repository layout

```
.
├── src/patpy/                     # the patpy library
│   ├── pp/  tl/  pl/  datasets/   # Public API: pp (preprocessing), tl (tools), pl (plots)
│   └── skills/                    # SKILL.md files — see "Skills" below
├── mcp/                           # the patpy-mcp standalone subproject
│   ├── pyproject.toml             # patpy-mcp's own packaging (separate from the parent)
│   ├── meta.yaml                  # BioContextAI Registry: Schema.org metadata
│   ├── mcp.json                   # BioContextAI Registry: MCP client config snippet ({ mcpServers: { ... } })
│   ├── Dockerfile                 # slim deploy image (build context = repo root for shared LICENSE)
│   ├── src/patpy_mcp/             # package code (cookiecutter layout)
│   │   ├── main.py                # click CLI entrypoint (run_app)
│   │   ├── mcp.py                 # module-level FastMCP instance
│   │   ├── cache.py               # ~/.cache/patpy-mcp/ layout + sidecars
│   │   ├── sources/cellxgene/     # REST client for CellxGene Discover
│   │   └── tools/_<name>.py       # ONE tool per file, decorated with @mcp.tool
│   └── tests/                     # patpy-mcp's own test suite
├── tests/                         # patpy's test suite (unrelated to mcp/tests/)
├── docs/                          # Sphinx docs (incl. docs/mcp.md)
├── pyproject.toml                 # patpy package config
├── README.md                      # human-facing
└── AGENTS.md                      # this file
```

When in doubt: edits to anything under `src/patpy/` belong to the `patpy` package; anything under `mcp/` belongs to `patpy-mcp`. Their dependencies, tests, and release pipelines are intentionally separate.

## Environments and tooling

The repo uses **`uv`** for the patpy-mcp venv and **mamba/conda** for the patpy main env (because patpy has many heavyweight scientific deps).

- The patpy-mcp subproject has its own venv at `.venv-patpy-mcp/`, created with:
  ```bash
  uv venv .venv-patpy-mcp --python 3.12
  source .venv-patpy-mcp/bin/activate
  uv pip install -e "./mcp[test]"
  ```
- For the main patpy library, use mamba/conda envs as the user prefers; do **not** install patpy and patpy-mcp into the same env unless you have a reason to.

## Running tests

Two separate suites; run them in their own roots:

```bash
# patpy-mcp
cd mcp && pytest                          # 20 tests, runs in <5 s, fully offline

# patpy
cd <repo root> && pytest                  # full patpy suite (needs the patpy env)
```

Always run `pytest` from inside `mcp/` for patpy-mcp work — its `pyproject.toml` sets `pythonpath = "src"` so imports resolve correctly only from there.

## Coding conventions

- **Formatter / linter**: ruff. Both `pyproject.toml` files declare ruff config; respect `line-length = 120`.
- **Public API discipline**: only symbols re-exported from `patpy.__init__`, `patpy.pp`, `patpy.tl`, `patpy.pl`, `patpy.datasets` are public. Anything starting with `_` is private. Do not call private symbols from new code, tests, or skills.
- **Type hints**: required on all new functions; use `from __future__ import annotations` at the top of new modules.
- **Tests for new tools** (in `mcp/`): one tool per file, mock HTTP with the hand-rolled `_FakeSession` pattern in `mcp/tests/test_cellxgene_discover.py` — do not introduce `responses` or `pytest-httpx` dependencies.

## Skills (where to look first when given a task)

`src/patpy/skills/` is an index of **task-specific guidance** the agent should consult before writing code that uses patpy or patpy-mcp. Every subdirectory has a `SKILL.md` whose YAML frontmatter declares `name:` + `description:`; read the description to decide whether to load the body.

**Never guess the patpy API — always consult the relevant skill first.** The model's training-time priors on patpy are weak; the skills capture conventions and gotchas that are not obvious from the source alone.

| Task at hand | Read this skill |
| --- | --- |
| Find / download a public single-cell dataset (CellxGene Discover) | [`src/patpy/skills/cellxgene/SKILL.md`](src/patpy/skills/cellxgene/SKILL.md) |
| QC, filter samples / cell groups, prepare AnnData | [`src/patpy/skills/preprocessing/SKILL.md`](src/patpy/skills/preprocessing/SKILL.md) |
| Build a sample×sample distance matrix from cells | [`src/patpy/skills/sample_representation/SKILL.md`](src/patpy/skills/sample_representation/SKILL.md) |
| Predict donor-level labels from per-cell features | [`src/patpy/skills/supervised_methods/SKILL.md`](src/patpy/skills/supervised_methods/SKILL.md) |
| Score a representation against metadata (kNN, silhouette, distance test, persistence) | [`src/patpy/skills/evaluation/SKILL.md`](src/patpy/skills/evaluation/SKILL.md) |
| Volcano / heatmap plots | [`src/patpy/skills/plotting/SKILL.md`](src/patpy/skills/plotting/SKILL.md) |
| Synthetic data with controlled perturbations | [`src/patpy/skills/datasets/SKILL.md`](src/patpy/skills/datasets/SKILL.md) |

Top-level index with cross-cutting pitfalls and the `pp`/`tl`/`pl` mental model: [`src/patpy/skills/SKILL.md`](src/patpy/skills/SKILL.md).

## MCP server (`patpy-mcp`)

`patpy-mcp` exposes nine tools that any MCP-capable agent can call:

| Tool | Purpose |
| --- | --- |
| `list_sources` / `describe_source` | Discover which data sources are enabled. |
| `cellxgene_search_datasets` | Search CellxGene Discover by disease / tissue / organism / assay. |
| `cellxgene_get_dataset` | Full per-dataset metadata + asset list. |
| `cellxgene_list_collections` / `cellxgene_get_collection` | Collection (publication) browsing. |
| `cellxgene_list_disease_terms` / `cellxgene_list_tissue_terms` | Distinct ontology terms (label + ID). |
| `cellxgene_download_dataset` | Stream a dataset asset to `$PATPY_MCP_CACHE` with SHA-256 + sidecar. |

Run it with `uvx patpy-mcp` or `patpy-mcp` (after `pip install patpy-mcp` or after activating `.venv-patpy-mcp`). Configure clients with:

```json
{ "mcpServers": { "patpy": { "command": "patpy-mcp" } } }
```

Detailed user-facing docs: [`docs/mcp.md`](docs/mcp.md). Subproject README: [`mcp/README.md`](mcp/README.md).

### Adding a new MCP tool

1. Create `mcp/src/patpy_mcp/tools/_<your_tool>.py` with one `@mcp.tool`-decorated function. The function name **becomes** the tool name; no `name=` argument needed.
2. Re-export it from `mcp/src/patpy_mcp/tools/__init__.py` (the side-effect import that `main.py`'s `from .tools import *` relies on).
3. Add a regression test under `mcp/tests/`.
4. Update [`src/patpy/skills/cellxgene/SKILL.md`](src/patpy/skills/cellxgene/SKILL.md) (or write a new skill) so future agents know when to call your tool.

### Hard-won gotchas (do not regress)

- **fastmcp ≥ 2** uses `on_duplicate="error"`, **not** the pre-2.0 `on_duplicate_tools=`. We use `on_duplicate="error"` in `mcp/src/patpy_mcp/mcp.py` so accidental tool-name collisions blow up at import time.
- **CellxGene Curation API has no flat `/datasets/{id}` endpoint.** Per-dataset metadata only lives at `/collections/{cid}/datasets/{dsid}`. `DiscoverClient.get_dataset_raw` resolves the parent collection from the cached `/datasets` list. There is a regression test (`test_get_dataset_uses_nested_collection_endpoint`) — keep it green.
- **`fastmcp.Client(mcp)` is the right way to drive the server in tests** (in-memory transport, no subprocess). See `try_mcp.py` at the repo root for a runnable demo.
- **`asyncio.run(mcp.list_tools())`** is how you introspect registered tools in fastmcp 2.x. There is no synchronous `get_tools()`.
- **The `LICENSE` file is shared.** `mcp/pyproject.toml` references `../LICENSE` from the parent repo; do not duplicate it under `mcp/`.

## When you should NOT modify something

- `src/patpy/skills/SKILL.md`'s frontmatter `description:` — changing it changes how the agent routes; coordinate with the maintainers.
- `mcp/meta.yaml` — this is the BioContextAI Registry submission. Edits propagate to a separate registry repository.
- `.github/workflows/release*.yaml` — releases are triggered by tag prefix (`v*` for patpy, `patpy-mcp-v*` for patpy-mcp); do not change the trigger conditions casually.
- `pyproject.toml` `version = "..."` — the release pipeline drives version bumps; don't hand-edit.

## Pull request expectations

1. Both test suites pass (`pytest` in repo root *and* in `mcp/`).
2. New behaviour comes with tests. Bug fixes come with regression tests.
3. New tools / functions have type hints and docstrings — those docstrings are what the LLM reads when picking the tool.
4. Update the relevant `SKILL.md` if you changed the public API surface or added a new common workflow.
5. Keep `patpy-mcp` self-contained — do **not** add `patpy` as a dependency of `patpy-mcp` unless you're shipping a tool that genuinely needs the patpy library at runtime.

---

If anything in this file is out of date, fix it as part of the PR that made it stale. Future agents will thank you.
