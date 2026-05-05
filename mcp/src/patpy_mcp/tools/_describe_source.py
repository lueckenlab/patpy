"""``describe_source`` tool: expand a single source descriptor."""

from __future__ import annotations

from typing import Any

from patpy_mcp.mcp import mcp
from patpy_mcp.sources import AVAILABLE_SOURCES


@mcp.tool
def describe_source(name: str) -> dict[str, Any]:
    """Return the descriptor for a single registered data source.

    Parameters
    ----------
    name
        Source identifier as returned by ``list_sources`` (e.g.
        ``"cellxgene"``).

    Returns
    -------
    dict
        ``{name, description, capabilities, homepage}``.

    Raises
    ------
    ValueError
        If ``name`` is not a registered source.
    """
    for source in AVAILABLE_SOURCES:
        if source.name == name:
            return {
                "name": source.name,
                "description": source.description,
                "capabilities": list(source.capabilities),
                "homepage": source.homepage,
            }
    available = sorted(s.name for s in AVAILABLE_SOURCES)
    raise ValueError(f"Unknown data source {name!r}. Available sources: {available}")
