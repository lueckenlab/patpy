# MCP server

patpy ships an MCP (Model Context Protocol) server, `patpy-mcp`, that lets any
MCP-capable LLM agent search and download single-cell datasets from public
registries on your behalf.

The server is part of the
[BioContextAI Registry](https://biocontext.ai/registry) and is designed to be
**combined** with sibling registry servers rather than duplicate them:

| Task                              | Server                                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Dataset search and download       | **patpy-mcp** (this server)                                                                             |
| Census slice queries (TileDB-SOMA) | [`MaxMLang/cxg-census-mcp`](https://github.com/MaxMLang/cxg-census-mcp)                                 |
| AnnData inspection                | [`biocontext-ai/anndata-mcp`](https://github.com/biocontext-ai/anndata-mcp)                             |
| scanpy / decoupler / cellrank execution | [`scmcphub/scmcp`](https://github.com/scmcphub/scmcp)                                              |

All four servers speak MCP over stdio, so any compliant agent can run them in
parallel and chain their outputs.

## Install and run

```bash
pip install 'patpy[mcp]'
patpy-mcp                       # launches the stdio MCP server
```

Or via Docker:

```bash
docker build -t patpy-mcp -f mcp/Dockerfile .
docker run --rm -i -v "$HOME/.cache/patpy-mcp:/data/cache" patpy-mcp
```

Downloaded files land under `~/.cache/patpy-mcp/datasets/cellxgene/<dataset_id>/`
(or `$PATPY_MCP_CACHE` if set), with a `<file>.meta.json` sidecar recording the
SHA-256, byte size, fetch timestamp, and source URL. Other MCP servers (notably
`anndata-mcp`) can read those files directly.

## Tools exposed

| Tool                              | Purpose                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `list_sources`                    | List dataset sources enabled in this server build.                                       |
| `describe_source`                 | Description and capabilities for one source.                                             |
| `cellxgene_search_datasets`       | Search CellxGene Discover by disease, tissue, organism, assay, or free-text title.       |
| `cellxgene_get_dataset`           | Full metadata for a dataset, including downloadable assets.                              |
| `cellxgene_list_collections`      | List CellxGene collections (publications), optionally filtered by free text.             |
| `cellxgene_get_collection`        | Full metadata for a collection, including its datasets.                                  |
| `cellxgene_list_disease_terms`    | Distinct disease ontology terms present in CellxGene (label + ontology ID).              |
| `cellxgene_list_tissue_terms`     | Distinct tissue ontology terms present in CellxGene.                                     |
| `cellxgene_download_dataset`      | Stream-download a dataset asset to the local cache, returning path / size / SHA-256.     |

Filter values for `disease`, `tissue`, and `assay` accept either ontology IDs
(e.g. `MONDO:0007254`) or labels (e.g. `breast carcinoma`).

## Connecting an agent

### Claude Desktop, Cursor, Claude Code (stdio config)

Add the three BioContextAI servers to your client's MCP config:

```json
{
  "mcpServers": {
    "patpy":   { "command": "patpy-mcp" },
    "cxg":     { "command": "uvx", "args": ["cxg-census-mcp"] },
    "anndata": { "command": "uvx", "args": ["anndata-mcp"] }
  }
}
```

### Local open-source agents (Llama, Mistral, ... via Ollama)

```bash
# Bring up Ollama with any chat-capable model:
ollama pull llama3.1:8b

# Use mcp-cli to chat with all three servers:
mcp-cli chat --servers patpy,cxg,anndata --provider ollama --model llama3.1
```

[Goose](https://github.com/block/goose) and
[Continue.dev](https://continue.dev/) accept the same `command: patpy-mcp`
declaration in their respective config files.

## Sample workflow

> *"Find breast cancer datasets in CellxGene with at least 50 k cells, list the
> top three by cell count, then download the smallest one with patpy and use
> anndata-mcp to summarise its obs columns."*

A capable agent will roughly do:

1. `cellxgene_list_disease_terms(prefix="breast")` to find the right ontology
   term, e.g. `MONDO:0007254`.
2. `cellxgene_search_datasets(disease=["MONDO:0007254"], min_cells=50000, limit=3)`.
3. `cellxgene_download_dataset(dataset_id=..., max_size_gb=5.0)`.
4. Forward the returned `local_path` to `anndata-mcp`'s inspection tools.

Once the dataset is on disk you can run patpy's sample-representation methods
(`patpy.tl.GloScope`, `patpy.tl.MOFA`, `patpy.tl.SCPoli`, ...) on it from a
notebook or script and compare against your own data.

## Cache layout

```
~/.cache/patpy-mcp/
├── datasets/
│   └── cellxgene/
│       └── <dataset_id>/
│           ├── <asset>.h5ad
│           └── <asset>.h5ad.meta.json
└── index/
    ├── cellxgene_collections.json   # 24 h TTL
    └── cellxgene_datasets.json      # 24 h TTL
```

Override with `$PATPY_MCP_CACHE`, or pass `out_dir=...` to
`cellxgene_download_dataset` for one-off destinations.
