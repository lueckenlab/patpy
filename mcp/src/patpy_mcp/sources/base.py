"""Abstractions shared by every patpy-mcp data source descriptor.

Tools are registered via the cookiecutter convention (one file per tool
under :mod:`patpy_mcp.tools` with ``@mcp.tool``), so a source no longer
needs to ``register(mcp)`` itself. The :class:`DataSource` dataclass
below is purely a metadata descriptor consumed by the cross-source
``list_sources`` / ``describe_source`` tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


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


@dataclass(frozen=True)
class DataSource:
    """Metadata descriptor for a registered data source."""

    name: str
    """Stable short identifier, used as a tool-name prefix (e.g. ``cellxgene``)."""

    description: str
    """One-paragraph summary of the source, surfaced via ``describe_source``."""

    capabilities: tuple[str, ...]
    """Short tags such as ``("search_datasets", "download_dataset")``."""

    homepage: str = ""
    """Optional human-facing URL to the underlying registry."""
