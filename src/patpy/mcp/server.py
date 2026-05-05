from __future__ import annotations

from .tools import (
    build_representation,
    dataset_summary,
    evaluate_representation,
    generate_plot,
    preprocess_dataset,
    run_supervised_prediction,
    simulate_dataset,
)


def create_server():
    """Create the optional FastMCP server wrapper for patpy tools."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("patpy-skills-mcp requires the optional MCP dependency. Install with `pip install patpy[mcp]`.") from exc

    server = FastMCP("patpy")
    server.tool()(dataset_summary)
    server.tool()(preprocess_dataset)
    server.tool()(build_representation)
    server.tool()(evaluate_representation)
    server.tool()(run_supervised_prediction)
    server.tool()(generate_plot)
    server.tool()(simulate_dataset)
    return server


def main() -> int:
    """Run the local stdio MCP server."""

    create_server().run()
    return 0
