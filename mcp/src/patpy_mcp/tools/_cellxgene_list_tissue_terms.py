"""``cellxgene_list_tissue_terms`` tool."""

from __future__ import annotations

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
def cellxgene_list_tissue_terms(prefix: str | None = None, limit: int = 200) -> list[dict[str, str]]:
    """List distinct tissue ontology terms present in CellxGene."""
    return discover_client.list_tissue_terms(prefix=prefix, limit=limit)
