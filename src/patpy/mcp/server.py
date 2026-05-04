"""FastMCP server for patpy dataset discovery.

Run with the ``patpy-mcp`` console script (registered in
``pyproject.toml``) or as a module::

    python -m patpy.mcp.server

The server speaks MCP over stdio, which any compliant agent (Claude
Desktop, Cursor, Goose, mcp-cli, ...) connects to via a small JSON
config snippet.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from patpy.mcp.sources import AVAILABLE_SOURCES

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("patpy.mcp")

SERVER_NAME = "patpy-mcp"
SERVER_INSTRUCTIONS = (
    "patpy MCP server for sample-level single-cell dataset discovery. "
    "Use 'list_sources' / 'describe_source' to discover available data sources, "
    "then call source-specific tools (e.g. 'cellxgene_search_datasets', "
    "'cellxgene_download_dataset'). For Census slice queries chain this server "
    "with MaxMLang/cxg-census-mcp; for AnnData inspection chain with "
    "biocontext-ai/anndata-mcp; downloaded files share a common cache so "
    "their absolute paths can be passed directly between servers."
)


def build_server() -> FastMCP:
    """Construct a FastMCP server with every available source registered.

    Imported lazily so the rest of the package can be imported without the
    optional ``mcp`` dependency installed (useful for unit-testing the
    underlying clients).
    """
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415
    except ImportError as err:
        raise ImportError(
            "The MCP extras are not installed. Run: pip install 'patpy[mcp]'"
        ) from err

    mcp = FastMCP(name=SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    @mcp.tool(name="list_sources")
    def list_sources() -> list[dict[str, Any]]:
        """List dataset sources enabled in this server build.

        Returns
        -------
        list of dict
            One entry per source with ``name``, ``description``, and ``capabilities``.
        """
        return [
            {
                "name": src.name,
                "description": src.description,
                "capabilities": list(src.capabilities),
            }
            for src in AVAILABLE_SOURCES
        ]

    @mcp.tool(name="describe_source")
    def describe_source(name: str) -> dict[str, Any]:
        """Return the description and capabilities of a single source by name.

        Parameters
        ----------
        name
            Source identifier as returned by ``list_sources`` (e.g. ``"cellxgene"``).
        """
        for src in AVAILABLE_SOURCES:
            if src.name == name:
                return {
                    "name": src.name,
                    "description": src.description,
                    "capabilities": list(src.capabilities),
                }
        known = ", ".join(sorted(s.name for s in AVAILABLE_SOURCES))
        raise ValueError(f"Unknown source '{name}'. Known sources: {known}")

    for src in AVAILABLE_SOURCES:
        try:
            src.register(mcp)
        except Exception:
            logger.exception("Failed to register source %r; continuing without it.", src.name)

    return mcp


def main() -> None:
    """Console entrypoint registered as ``patpy-mcp``."""
    logging.basicConfig(
        level=os.environ.get("PATPY_MCP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting %s with sources: %s", SERVER_NAME, [s.name for s in AVAILABLE_SOURCES])
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
