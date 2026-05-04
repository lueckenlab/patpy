"""Abstractions shared by every patpy MCP data source.

A *source* is a thin adapter around an external dataset registry
(CellxGene, HCA, GEO, ...). It exposes its functionality to MCP-capable
agents by registering tools on a FastMCP server. The protocol below is
intentionally minimal — sources are free to register additional
source-specific tools (and most will).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class DatasetSummary(TypedDict, total=False):
    """Compact dataset record returned by ``*_search_datasets`` tools."""

    source: str
    dataset_id: str
    title: str
    collection_id: str
    collection_name: str
    organism: list[str]
    disease: list[str]
    tissue: list[str]
    assay: list[str]
    cell_count: int
    donor_count: int
    primary_data: bool
    explorer_url: str


class DatasetMetadata(TypedDict, total=False):
    """Full dataset metadata returned by ``*_get_dataset``."""

    source: str
    dataset_id: str
    title: str
    description: str
    collection_id: str
    collection_name: str
    organism: list[str]
    disease: list[str]
    tissue: list[str]
    assay: list[str]
    cell_count: int
    donor_count: int
    schema_version: str
    assets: list[dict[str, Any]]
    explorer_url: str
    raw: dict[str, Any]


class DownloadResult(TypedDict):
    """Return type of ``*_download_dataset`` tools."""

    source: str
    dataset_id: str
    asset_id: str
    local_path: str
    size_bytes: int
    sha256: str
    cached: bool
    source_url: str


@runtime_checkable
class DataSource(Protocol):
    """Minimal protocol every source plugin must satisfy."""

    name: str
    """Stable short identifier, used as a tool-name prefix (e.g. ``cellxgene``)."""

    description: str
    """One-sentence human-readable summary, surfaced via ``describe_source``."""

    capabilities: tuple[str, ...]
    """Short tags such as ``("search", "download", "ontology_terms")``."""

    def register(self, mcp: FastMCP) -> None:
        """Register all source-specific MCP tools on ``mcp``.

        Implementations should namespace tool names by ``self.name`` to
        avoid clashes when multiple sources are loaded into the same
        server.
        """
