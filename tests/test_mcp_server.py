"""Smoke tests for the patpy MCP server wiring.

The tests skip cleanly when the optional ``mcp`` extra is not installed,
so the rest of the suite remains green for users who only install patpy
core.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="patpy[mcp] extras are required for these tests.")
pytest.importorskip("requests", reason="patpy[mcp] extras are required for these tests.")

from patpy.mcp.server import SERVER_NAME, build_server  # noqa: E402
from patpy.mcp.sources import AVAILABLE_SOURCES  # noqa: E402


EXPECTED_GENERIC_TOOLS = {"list_sources", "describe_source"}
EXPECTED_CELLXGENE_TOOLS = {
    "cellxgene_search_datasets",
    "cellxgene_get_dataset",
    "cellxgene_list_collections",
    "cellxgene_get_collection",
    "cellxgene_list_disease_terms",
    "cellxgene_list_tissue_terms",
    "cellxgene_download_dataset",
}


def _registered_tool_names(server) -> set[str]:
    """Pull the set of registered tool names from a FastMCP instance."""
    if hasattr(server, "_tool_manager"):
        manager = server._tool_manager
        if hasattr(manager, "_tools"):
            return set(manager._tools.keys())
        if hasattr(manager, "list_tools"):
            tools = manager.list_tools()
            return {getattr(t, "name", None) or t["name"] for t in tools}
    if hasattr(server, "list_tools"):
        return {getattr(t, "name", None) or t["name"] for t in server.list_tools()}
    raise AssertionError("Could not introspect tool names from FastMCP instance.")


def test_build_server_exposes_expected_tools() -> None:
    server = build_server()
    assert server.name == SERVER_NAME
    names = _registered_tool_names(server)
    missing = (EXPECTED_GENERIC_TOOLS | EXPECTED_CELLXGENE_TOOLS) - names
    assert not missing, f"Missing tools: {sorted(missing)}; registered: {sorted(names)}"


def test_cellxgene_source_is_registered() -> None:
    sources = {s.name for s in AVAILABLE_SOURCES}
    assert "cellxgene" in sources
