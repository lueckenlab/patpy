"""Module-level FastMCP instance shared by every tool in :mod:`patpy_mcp.tools`.

The cookiecutter convention is to define the server as a module-level
singleton so that every ``tools/_<toolname>.py`` file can import it and
attach itself with ``@mcp.tool``. Setting ``on_duplicate="error"``
catches accidental name clashes at import time.
"""

from fastmcp import FastMCP

mcp: FastMCP = FastMCP(
    name="patpy-mcp",
    instructions=(
        "patpy MCP server for sample-level single-cell dataset discovery. "
        "Use 'list_sources' / 'describe_source' to discover available data sources, "
        "then call source-specific tools (e.g. 'cellxgene_search_datasets', "
        "'cellxgene_download_dataset'). For Census slice queries chain this server "
        "with MaxMLang/cxg-census-mcp; for AnnData inspection chain with "
        "biocontext-ai/anndata-mcp; downloaded files share a common cache so their "
        "absolute paths can be passed directly between servers."
    ),
    on_duplicate="error",
)
