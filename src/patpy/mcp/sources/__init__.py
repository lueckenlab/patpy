"""Pluggable data sources exposed through the patpy MCP server.

Each source implements the :class:`patpy.mcp.sources.base.DataSource`
protocol and registers a set of MCP tools when handed a FastMCP server
instance. New sources (HCA, GEO, Single Cell Portal, ...) plug in by
adding an entry to :data:`AVAILABLE_SOURCES`.
"""

from __future__ import annotations

from patpy.mcp.sources.base import DataSource
from patpy.mcp.sources.cellxgene.source import CellxGeneSource

AVAILABLE_SOURCES: tuple[DataSource, ...] = (CellxGeneSource(),)

__all__ = ["AVAILABLE_SOURCES", "CellxGeneSource", "DataSource"]
