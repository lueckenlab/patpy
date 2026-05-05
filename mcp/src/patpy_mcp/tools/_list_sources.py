"""``list_sources`` tool: enumerate registered data sources."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources import AVAILABLE_SOURCES


@mcp.tool
def list_sources() -> list[dict[str, Any]]:
    """List the data sources this patpy-mcp build can query.

    Each entry has ``name``, ``description``, ``capabilities``, and
    ``homepage``. Tools for a given source are namespaced with that
    name (e.g. the ``cellxgene`` source provides ``cellxgene_*`` tools).
    """
    return [
        {
            "name": source.name,
            "description": source.description,
            "capabilities": list(source.capabilities),
            "homepage": source.homepage,
        }
        for source in AVAILABLE_SOURCES
    ]
