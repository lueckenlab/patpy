"""Tiny no-Node, no-LLM way to drive patpy-mcp.

Run with the patpy-mcp venv active::

    source .venv-patpy-mcp/bin/activate
    export PATPY_MCP_CACHE="$PWD/.cache-patpy-mcp-test"
    python try_mcp.py

It uses fastmcp's in-memory transport, so no stdio server has to be
running on the side. Edit the ``demo`` coroutine to call any of the
nine tools.
"""

from __future__ import annotations

import asyncio
import json

import patpy_mcp.tools  # noqa: F401  side-effect: register every tool
from fastmcp import Client
from patpy_mcp.mcp import mcp


def _summarise(call_result) -> object:
    """Fastmcp 2.x returns ``CallToolResult`` whose ``.data`` is the parsed payload."""
    data = getattr(call_result, "data", None)
    return data if data is not None else getattr(call_result, "content", call_result)


async def demo() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print(f"=== Server advertises {len(tools)} tools ===")
        for t in sorted(tools, key=lambda t: t.name):
            first_line = (t.description or "").split("\n", 1)[0][:70]
            print(f"  {t.name:<32} {first_line}")

        print("\n=== list_sources ===")
        out = await client.call_tool("list_sources", {})
        print(json.dumps(_summarise(out), indent=2)[:500], "…")

        print("\n=== cellxgene_list_disease_terms(prefix='lung') ===")
        out = await client.call_tool("cellxgene_list_disease_terms", {"prefix": "lung", "limit": 5})
        for t in _summarise(out):
            print(f"  {t['ontology_term_id']:<20} {t['label']}")

        print("\n=== cellxgene_search_datasets(disease=['lung adenocarcinoma'], min_cells=50000, limit=3) ===")
        out = await client.call_tool(
            "cellxgene_search_datasets",
            {
                "disease": ["lung adenocarcinoma"],
                "min_cells": 50_000,
                "limit": 3,
            },
        )
        for d in _summarise(out):
            print(f"  {d['dataset_id']}  cells={d['cell_count']:>10,}  '{d['title'][:55]}'")


if __name__ == "__main__":
    asyncio.run(demo())
