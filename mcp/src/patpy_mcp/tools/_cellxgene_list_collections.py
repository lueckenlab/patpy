"""``cellxgene_list_collections`` tool."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
def cellxgene_list_collections(query: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
    """List CellxGene collections (publications), optionally filtered by ``query``.

    Each collection bundles one or more datasets that came out of a single
    study. Use ``cellxgene_get_collection`` to drill into one.
    """
    return discover_client.list_collections(query=query, limit=limit)
