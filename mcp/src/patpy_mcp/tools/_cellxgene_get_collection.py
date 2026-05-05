"""``cellxgene_get_collection`` tool."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
def cellxgene_get_collection(collection_id: str) -> dict[str, Any]:
    """Return full metadata for a CellxGene collection, including its datasets."""
    return discover_client.get_collection(collection_id)
