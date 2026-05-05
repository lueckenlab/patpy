"""CellxGene data source for patpy-mcp.

Wraps the public CellxGene Discover Curation REST API at
``https://api.cellxgene.cziscience.com/curation/v1/``. Census slice
queries are intentionally **not** implemented here -- those are covered
by the existing ``MaxMLang/cxg-census-mcp`` server in the BioContextAI
Registry and we recommend running it alongside.

The actual MCP tools are defined in :mod:`patpy_mcp.tools` (one file per
tool); this module only exposes the data-source descriptor and the
shared :class:`DiscoverClient` instance used by every cellxgene tool.
"""

from __future__ import annotations

from patpy_mcp.sources.base import DataSource
from patpy_mcp.sources.cellxgene.discover import DiscoverClient

CELLXGENE_SOURCE = DataSource(
    name="cellxgene",
    description=(
        "CellxGene Discover (https://cellxgene.cziscience.com): public catalogue of "
        "single-cell datasets. This source provides dataset-level search and download. "
        "For Census slice queries use MaxMLang/cxg-census-mcp; for AnnData inspection "
        "use biocontext-ai/anndata-mcp."
    ),
    capabilities=(
        "search_datasets",
        "list_collections",
        "ontology_terms",
        "download_dataset",
    ),
    homepage="https://cellxgene.cziscience.com",
)

discover_client: DiscoverClient = DiscoverClient()
"""Shared :class:`DiscoverClient` used by every ``cellxgene_*`` tool.

Re-using a single ``requests.Session`` across tool invocations lets the
underlying connection pool kick in for repeated calls inside a single
agent turn (e.g. searching, then immediately downloading).
"""

__all__ = ["CELLXGENE_SOURCE", "DiscoverClient", "discover_client"]
