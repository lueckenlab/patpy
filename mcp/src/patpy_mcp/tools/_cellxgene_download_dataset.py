"""``cellxgene_download_dataset`` tool."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources.cellxgene import discover_client


@mcp.tool
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
    return discover_client.download_dataset(
        dataset_id=dataset_id,
        asset_format=asset_format,
        out_dir=out_dir,
        max_size_gb=max_size_gb,
        force=force,
    )
