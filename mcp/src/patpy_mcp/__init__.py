"""patpy-mcp: MCP server for single-cell dataset discovery.

Built with the BioContextAI MCP server cookiecutter conventions
(https://github.com/biocontext-ai/mcp-server-cookiecutter): one tool per
file under :mod:`patpy_mcp.tools`, a shared :data:`patpy_mcp.mcp.mcp`
FastMCP instance, and a click-based CLI in :mod:`patpy_mcp.main`.
"""

from importlib.metadata import PackageNotFoundError, version

from patpy_mcp.main import run_app
from patpy_mcp.mcp import mcp

try:
    __version__ = version("patpy-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "mcp", "run_app"]


if __name__ == "__main__":
    run_app()
