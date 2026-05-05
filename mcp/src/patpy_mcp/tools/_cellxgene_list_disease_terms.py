"""``cellxgene_list_disease_terms`` tool."""

from __future__ import annotations

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
def cellxgene_list_disease_terms(prefix: str | None = None, limit: int = 200) -> list[dict[str, str]]:
    """List distinct disease ontology terms present in CellxGene.

    Useful before calling ``cellxgene_search_datasets`` so the agent can map a
    free-text query like "breast cancer" to a precise ontology term such as
    ``MONDO:0007254`` ("breast carcinoma").

    Parameters
    ----------
    prefix
        Optional case-insensitive prefix on either label or ontology ID.
    limit
        Maximum number of terms to return.
    """
    return discover_client.list_disease_terms(prefix=prefix, limit=limit)
