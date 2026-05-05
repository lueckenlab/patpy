"""``cellxgene_get_dataset`` tool."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
def cellxgene_get_dataset(dataset_id: str) -> dict[str, Any]:
    """Return full metadata for a CellxGene dataset, including downloadable assets.

    Parameters
    ----------
    dataset_id
        Dataset UUID as returned by ``cellxgene_search_datasets``.
    """
    return discover_client.get_dataset(dataset_id)
