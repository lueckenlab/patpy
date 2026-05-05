"""Pluggable data sources exposed through patpy-mcp.

Each source is a lightweight :class:`patpy_mcp.sources.base.DataSource`
descriptor (name + description + capabilities) that the
``list_sources`` and ``describe_source`` tools surface to the agent.
The actual MCP tools are defined as one-tool-per-file modules under
:mod:`patpy_mcp.tools` following the BioContextAI cookiecutter
convention.
"""

from __future__ import annotations

from patpy_mcp.sources.base import DataSource
from patpy_mcp.sources.cellxgene import CELLXGENE_SOURCE

AVAILABLE_SOURCES: tuple[DataSource, ...] = (CELLXGENE_SOURCE,)

__all__ = ["AVAILABLE_SOURCES", "CELLXGENE_SOURCE", "DataSource"]
