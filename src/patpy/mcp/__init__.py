"""Model Context Protocol (MCP) server for sample-level dataset discovery.

This subpackage exposes patpy's dataset-discovery tools through the
Model Context Protocol so that any MCP-capable LLM agent (Claude
Desktop, Cursor, Goose, mcp-cli + Ollama, ...) can search and download
single-cell datasets from public registries (starting with CellxGene
Discover).

The server is registry-compatible with the BioContextAI Registry
(https://biocontext.ai/registry) and is designed to chain with
``MaxMLang/cxg-census-mcp`` for Census slice queries and
``biocontext-ai/anndata-mcp`` for AnnData inspection.

Install with::

    pip install patpy[mcp]

and launch with::

    patpy-mcp
"""

from patpy.mcp.server import build_server, main

__all__ = ["build_server", "main"]
