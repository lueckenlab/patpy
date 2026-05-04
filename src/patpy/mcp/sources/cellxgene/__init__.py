"""CellxGene data source for the patpy MCP server.

Wraps the public CellxGene Discover Curation REST API at
``https://api.cellxgene.cziscience.com/curation/v1/`` to expose search,
metadata, and download capabilities through MCP tools.

Census slice queries are intentionally **not** implemented here — those
are covered by the existing ``MaxMLang/cxg-census-mcp`` server in the
BioContextAI Registry and we recommend running it alongside.
"""

from patpy.mcp.sources.cellxgene.source import CellxGeneSource

__all__ = ["CellxGeneSource"]
