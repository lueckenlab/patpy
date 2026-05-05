"""Smoke tests for the patpy-mcp CLI and tool registry.

The tests intentionally avoid spinning up the actual MCP transport;
they only exercise (a) the click ``--version`` short-circuit and (b)
the side-effect tool registration triggered by importing
:mod:`patpy_mcp.tools`.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastmcp", reason="fastmcp is required for these tests.")
pytest.importorskip("click", reason="click is required for these tests.")

import click.testing

# Importing main has the side-effect of registering every tool because
# main.py performs ``from .tools import *``.
from patpy_mcp import __version__
from patpy_mcp.main import run_app
from patpy_mcp.mcp import mcp
from patpy_mcp.sources import AVAILABLE_SOURCES

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
EXPECTED_TOOLS = EXPECTED_GENERIC_TOOLS | EXPECTED_CELLXGENE_TOOLS


def _registered_tool_names() -> set[str]:
    """Pull registered tool names off the module-level FastMCP instance.

    ``fastmcp`` 2.x exposes the public, async ``list_tools()`` coroutine
    that returns a list of :class:`fastmcp.tools.tool.FunctionTool`
    objects (each carrying a ``.name``). We run it synchronously here
    because the FastMCP server is not actually started by the test --
    we only want the in-process registry.
    """
    if hasattr(mcp, "list_tools"):
        tools = asyncio.run(mcp.list_tools())
        return {getattr(t, "name", None) or t["name"] for t in tools}
    raise AssertionError("Could not introspect tool names from the FastMCP instance.")


def test_server_metadata() -> None:
    """The server should advertise the agreed name."""
    assert mcp.name == "patpy-mcp"


def test_all_expected_tools_registered() -> None:
    """Importing ``patpy_mcp.main`` must register the full tool surface."""
    names = _registered_tool_names()
    missing = EXPECTED_TOOLS - names
    assert not missing, f"Missing tools: {sorted(missing)}; registered: {sorted(names)}"


def test_cellxgene_source_listed() -> None:
    sources = {s.name for s in AVAILABLE_SOURCES}
    assert "cellxgene" in sources


def test_cli_version_flag_exits_zero() -> None:
    """``patpy-mcp --version`` should print the package version and exit 0."""
    runner = click.testing.CliRunner()
    result = runner.invoke(run_app, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_cli_help_includes_transport_options() -> None:
    """The CLI surface should match the BioContextAI cookiecutter contract."""
    runner = click.testing.CliRunner()
    result = runner.invoke(run_app, ["--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--transport", "--port", "--host", "--env", "--version"):
        assert flag in result.output, f"missing flag {flag} in --help output"
