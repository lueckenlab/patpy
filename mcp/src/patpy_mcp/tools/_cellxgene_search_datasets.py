"""``cellxgene_search_datasets`` tool."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
def cellxgene_search_datasets(
    query: str | None = None,
    disease: list[str] | None = None,
    tissue: list[str] | None = None,
    organism: str = "Homo sapiens",
    assay: list[str] | None = None,
    min_cells: int | None = None,
    limit: int = 25,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Search CellxGene Discover for datasets matching the given filters.

    Use this to discover public single-cell datasets by disease, tissue,
    organism, or assay. ``disease`` / ``tissue`` / ``assay`` items can be
    either ontology IDs (e.g. ``"MONDO:0007254"``) or labels
    (e.g. ``"breast carcinoma"``); both are matched case-insensitively.

    Parameters
    ----------
    query
        Free-text substring matched against dataset titles.
    disease, tissue, assay
        Lists of ontology terms or labels; a dataset matches if any of its
        annotations matches any of the supplied values.
    organism
        Single organism label, defaults to ``"Homo sapiens"``. Pass ``""`` or
        ``None`` to disable the organism filter.
    min_cells
        Drop datasets with fewer than this many cells.
    limit, offset
        Standard pagination controls (client-side).

    Returns
    -------
    list of dict
        One :class:`patpy_mcp.sources.base.DatasetSummary` per matching dataset,
        with an ``explorer_url`` link to the public CellxGene viewer.
    """
    return discover_client.search_datasets(
        query=query,
        disease=disease,
        tissue=tissue,
        organism=organism or None,
        assay=assay,
        min_cells=min_cells,
        limit=limit,
        offset=offset,
    )
