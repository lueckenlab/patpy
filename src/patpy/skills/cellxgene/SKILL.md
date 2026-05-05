---
name: cellxgene-dataset-discovery
description: Find and download single-cell datasets from CellxGene Discover (e.g. "breast cancer", "lung adenocarcinoma", "kidney atlas") through the patpy-mcp server, then hand the file off to patpy / anndata-mcp for downstream analysis. Use when the user asks for a public single-cell dataset by disease, tissue, organism, or assay.
---

# CellxGene dataset discovery via patpy-mcp

`patpy` ships a separate MCP server, **`patpy-mcp`** ([repo](https://github.com/lueckenlab/patpy/tree/main/mcp)), that wraps the public [CellxGene Discover Curation API](https://api.cellxgene.cziscience.com/curation/v1/). It is the recommended way to fetch public single-cell data for a `patpy` workflow, because the downloaded files land in a shared cache that other BioContextAI servers (notably `biocontext-ai/anndata-mcp`) can read by absolute path.

## When to use

- The user names a disease / tissue / organism and wants to compare against public single-cell data ("find me breast cancer datasets", "any IPF lung atlas?").
- The user has their own AnnData and wants a public reference cohort to evaluate their representation against.
- The user wants to enumerate ontology terms actually present in CellxGene before searching.

Do **not** use this skill for:

- Cell-level slice queries across the integrated atlas → route to [`MaxMLang/cxg-census-mcp`](https://github.com/MaxMLang/cxg-census-mcp).
- Inspecting the contents of a downloaded `.h5ad` (obs/var columns, layers) → route to [`biocontext-ai/anndata-mcp`](https://github.com/biocontext-ai/anndata-mcp).
- Computing sample representations from cells → that's [`../sample_representation/SKILL.md`](../sample_representation/SKILL.md), driven from Python.

## Tool catalog (MCP tools exposed by `patpy-mcp`)

| Tool | Purpose |
|---|---|
| `list_sources()` | Enumerate enabled data sources (currently just `cellxgene`). |
| `describe_source(name)` | Capabilities and homepage for one source. |
| `cellxgene_list_disease_terms(prefix=None, limit=200)` | Distinct disease ontology terms (label + MONDO id). |
| `cellxgene_list_tissue_terms(prefix=None, limit=200)` | Distinct tissue ontology terms (label + UBERON id). |
| `cellxgene_list_collections(query=None, limit=25)` | Paginated study/publication list. |
| `cellxgene_get_collection(collection_id)` | Full collection metadata + dataset summaries. |
| `cellxgene_search_datasets(query=None, disease=None, tissue=None, organism="Homo sapiens", assay=None, min_cells=None, limit=25, offset=0)` | Filtered dataset list (client-side filters; cached 24 h). |
| `cellxgene_get_dataset(dataset_id)` | Full per-dataset metadata, including downloadable assets. |
| `cellxgene_download_dataset(dataset_id, asset_format="H5AD", out_dir=None, max_size_gb=10.0, force=False)` | Stream-download to the cache; returns `local_path`, `size_bytes`, `sha256`, `cached`, `source_url`. |

`disease`, `tissue`, `assay` accept either ontology IDs (`MONDO:0007254`) or labels (`breast carcinoma`); both are matched case-insensitively.

## Canonical workflow

Always follow this order — each step is cheap and protects you from wasted bandwidth:

1. **Map free text to an ontology term.**
   `cellxgene_list_disease_terms(prefix="breast")` → `MONDO:0007254` ("breast cancer").
   This guarantees the downstream filter works even if the user wrote "BRCA" or "breast carcinoma".

2. **Search with bounded arguments.**
   `cellxgene_search_datasets(disease=["MONDO:0007254"], min_cells=50_000, limit=10)`.
   Always pass `min_cells` to drop tiny samples, and keep `limit` ≤ 25.

3. **Inspect before downloading.**
   `cellxgene_get_dataset(dataset_id=...)` — check `assets[*].filesize`. If the only H5AD is e.g. 8 GB and the user didn't ask for that, raise it with the user instead of blasting the cache.

4. **Download with a size cap.**
   `cellxgene_download_dataset(dataset_id=..., max_size_gb=2.0)`.
   The default cap is 10 GB; raise it explicitly only after step 3.

5. **Hand the path off, don't re-read it here.**
   The return dict contains `local_path`. Pass that absolute path to `biocontext-ai/anndata-mcp`'s tools (or open it with `anndata.read_h5ad(local_path, backed="r")` from a notebook). `patpy-mcp` deliberately stops at download.

## Minimal walkthrough (agent-driven)

```text
User:  "Find me breast cancer datasets in CellxGene with at least 50 k cells, smallest one first."

Agent:
  1. cellxgene_list_disease_terms(prefix="breast")
     → [..., {"ontology_term_id": "MONDO:0007254", "label": "breast cancer"}, ...]

  2. cellxgene_search_datasets(disease=["MONDO:0007254"], min_cells=50_000, limit=5)
     → [{dataset_id: "f12ab0e6-…", cell_count: 10_957, title: "HTAPP-…"}, ...]

  3. cellxgene_get_dataset(dataset_id="f12ab0e6-…")
     → assets[0].filetype="H5AD", filesize=62_732_290  (≈60 MB, safe)

  4. cellxgene_download_dataset(dataset_id="f12ab0e6-…", max_size_gb=0.2)
     → local_path="/.../cellxgene/f12ab0e6-…/h5ad", sha256="c09c…", cached=False
```

After step 4 you can chain into `patpy.tl.Pseudobulk` (Python) or `anndata-mcp` tools (MCP) — see "Related skills" below.

## Common pitfalls

- **The CellxGene Curation API has no flat `/datasets/{id}` endpoint.** Per-dataset metadata only lives at `/collections/{cid}/datasets/{dsid}`. The MCP handles this transparently (it looks up the collection from the cached dataset list); never try to construct the URL yourself or you'll get `HTTP 404`.
- **Searching is client-side.** `cellxgene_search_datasets` filters in memory after fetching the cached `/datasets` payload (~9 MB). The first call in a fresh cache takes ~3–5 s; repeated calls are instant for 24 h.
- **`primary_data=False` rows.** Many search results are *integrated* derivative versions of the same primary cohort. If you only want primary studies, filter the result list on `primary_data == True`.
- **Cache layout matters.** Files land at `$PATPY_MCP_CACHE/datasets/cellxgene/<dataset_id>/<asset_id>` with a `.meta.json` sidecar (sha256, source_url, fetched_at). Re-running `cellxgene_download_dataset` with the same args returns `cached: true` in milliseconds. Pass the path directly to other tools — don't copy.
- **`max_size_gb` is a guard, not a quota.** It refuses *single-file* downloads above the threshold *before* streaming. Raising it lets the call through unconditionally; there's no automatic chunking.
- **Census ≠ Discover.** This skill targets the Discover REST API only, which serves *individual published datasets*. For unified queries across the entire integrated atlas (TileDB-SOMA), the user wants `MaxMLang/cxg-census-mcp` — that's a different MCP server.
- **Setup gotcha for testers.** The MCP requires `fastmcp ≥ 2`. The kwarg is `on_duplicate=` (not `on_duplicate_tools=` — that was the pre-2.0 spelling).

## How the user runs the server

`patpy-mcp` is its own PyPI package and is not pulled in by `pip install patpy`. Tell the user:

```bash
# Recommended (no install needed):
uvx patpy-mcp

# Or pinned:
pip install patpy-mcp
patpy-mcp
```

…then add it to their MCP client config:

```json
{ "mcpServers": { "patpy": { "command": "patpy-mcp" } } }
```

Cursor, Claude Desktop, mcp-cli + Ollama, and Goose all accept that snippet.

## Related skills

- After download, build a sample×sample distance matrix → [`../sample_representation/SKILL.md`](../sample_representation/SKILL.md).
- Score that matrix against `obs` metadata (e.g. `disease`, `treatment`) → [`../evaluation/SKILL.md`](../evaluation/SKILL.md).
- Filter the AnnData (sample size, cell-group size, count-data check) before representation → [`../preprocessing/SKILL.md`](../preprocessing/SKILL.md).
- Inject controlled perturbations into the downloaded data for benchmarking → [`../datasets/SKILL.md`](../datasets/SKILL.md).
