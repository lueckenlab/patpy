"""CellxGene MCP source plugin.

Translates the agent-facing tool surface (``cellxgene_search_datasets``,
``cellxgene_download_dataset``, ...) into calls on
:class:`patpy.mcp.sources.cellxgene.discover.DiscoverClient`.

The plugin is intentionally thin: docstrings and type hints declared
here become the JSON schema that the MCP client (and through it the
LLM) reads to choose tool arguments, so we keep them precise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from patpy.mcp.sources.cellxgene.discover import DiscoverClient

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass
class CellxGeneSource:
    """Implements :class:`patpy.mcp.sources.base.DataSource` for CellxGene Discover."""

    name: str = "cellxgene"
    description: str = (
        "CellxGene Discover (https://cellxgene.cziscience.com): public catalogue of "
        "single-cell datasets. This source provides dataset-level search and download. "
        "For Census slice queries use MaxMLang/cxg-census-mcp; for AnnData inspection "
        "use biocontext-ai/anndata-mcp."
    )
    capabilities: tuple[str, ...] = (
        "search_datasets",
        "list_collections",
        "ontology_terms",
        "download_dataset",
    )
    client: DiscoverClient = field(default_factory=DiscoverClient)

    def register(self, mcp: FastMCP) -> None:
        """Register every cellxgene_* tool on the FastMCP server."""
        client = self.client

        @mcp.tool(name="cellxgene_search_datasets")
        def cellxgene_search_datasets(  # noqa: D401  (description-only docstring)
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
                One :class:`patpy.mcp.sources.base.DatasetSummary` per matching dataset,
                with an ``explorer_url`` link to the public CellxGene viewer.
            """
            return client.search_datasets(
                query=query,
                disease=disease,
                tissue=tissue,
                organism=organism or None,
                assay=assay,
                min_cells=min_cells,
                limit=limit,
                offset=offset,
            )

        @mcp.tool(name="cellxgene_get_dataset")
        def cellxgene_get_dataset(dataset_id: str) -> dict[str, Any]:
            """Return full metadata for a CellxGene dataset, including downloadable assets.

            Parameters
            ----------
            dataset_id
                Dataset UUID as returned by ``cellxgene_search_datasets``.
            """
            return client.get_dataset(dataset_id)

        @mcp.tool(name="cellxgene_list_collections")
        def cellxgene_list_collections(query: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
            """List CellxGene collections (publications), optionally filtered by ``query``.

            Each collection bundles one or more datasets that came out of a single
            study. Use ``cellxgene_get_collection`` to drill into one.
            """
            return client.list_collections(query=query, limit=limit)

        @mcp.tool(name="cellxgene_get_collection")
        def cellxgene_get_collection(collection_id: str) -> dict[str, Any]:
            """Return full metadata for a CellxGene collection, including its datasets."""
            return client.get_collection(collection_id)

        @mcp.tool(name="cellxgene_list_disease_terms")
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
            return client.list_disease_terms(prefix=prefix, limit=limit)

        @mcp.tool(name="cellxgene_list_tissue_terms")
        def cellxgene_list_tissue_terms(prefix: str | None = None, limit: int = 200) -> list[dict[str, str]]:
            """List distinct tissue ontology terms present in CellxGene."""
            return client.list_tissue_terms(prefix=prefix, limit=limit)

        @mcp.tool(name="cellxgene_download_dataset")
        def cellxgene_download_dataset(
            dataset_id: str,
            asset_format: str = "H5AD",
            out_dir: str | None = None,
            max_size_gb: float = 10.0,
            force: bool = False,
        ) -> dict[str, Any]:
            """Stream-download a CellxGene dataset asset to local disk.

            Files land under ``$PATPY_MCP_CACHE`` (default
            ``~/.cache/patpy-mcp/datasets/cellxgene/<dataset_id>/``) with a
            ``<file>.meta.json`` sidecar recording size, SHA-256, and source URL.
            Re-running with the same arguments returns ``cached: true`` without
            re-downloading.

            Parameters
            ----------
            dataset_id
                Dataset UUID.
            asset_format
                One of the ``filetype`` values returned by ``cellxgene_get_dataset``
                (commonly ``"H5AD"`` or ``"RDS"``). Default ``"H5AD"``.
            out_dir
                Directory to write into; defaults to the patpy MCP cache.
            max_size_gb
                Refuse downloads whose advertised filesize exceeds this many GB.
                Raise it explicitly to fetch large objects.
            force
                If ``True``, re-download even when a cached copy exists.

            Returns
            -------
            dict
                ``{source, dataset_id, asset_id, local_path, size_bytes, sha256,
                cached, source_url}``. The ``local_path`` can be passed to other
                BioContextAI servers (e.g. ``biocontext-ai/anndata-mcp``) for
                downstream inspection.
            """
            return client.download_dataset(
                dataset_id=dataset_id,
                asset_format=asset_format,
                out_dir=out_dir,
                max_size_gb=max_size_gb,
                force=force,
            )
