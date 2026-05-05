"""Click-based CLI entrypoint for patpy-mcp.

Mirrors the BioContextAI cookiecutter ``main.py`` shape so that running
``uvx patpy-mcp`` (or ``patpy-mcp`` after ``pip install``) feels
identical to other registry servers. The ``from .tools import *``
import is intentional: it triggers the side-effect registration of
every ``@mcp.tool``-decorated function under ``patpy_mcp.tools``.
"""

from __future__ import annotations

import enum
import logging
import sys

import click

from .tools import *  # noqa: F401, F403  side-effect: registers every tool


class EnvironmentType(enum.Enum):
    """Runtime environment, controlling stdio vs. HTTP transport defaults."""

    PRODUCTION = enum.auto()
    DEVELOPMENT = enum.auto()


@click.command(name="run")
@click.option(
    "-t",
    "--transport",
    "transport",
    type=click.Choice(["stdio", "http", "sse"], case_sensitive=False),
    help="MCP transport. Defaults to 'stdio'.",
    default="stdio",
    envvar="MCP_TRANSPORT",
)
@click.option(
    "-p",
    "--port",
    "port",
    type=int,
    help="Port for HTTP/SSE transport. Defaults to 8000.",
    default=8000,
    envvar="MCP_PORT",
)
@click.option(
    "-h",
    "--host",
    "hostname",
    type=str,
    help="Hostname for HTTP/SSE transport. Defaults to '0.0.0.0'.",
    default="0.0.0.0",
    envvar="MCP_HOSTNAME",
)
@click.option(
    "-e",
    "--env",
    "environment",
    type=click.Choice(EnvironmentType, case_sensitive=False),
    default=EnvironmentType.DEVELOPMENT,
    envvar="MCP_ENVIRONMENT",
    help="MCP server environment. Defaults to 'development'.",
)
@click.option(
    "-v",
    "--version",
    "show_version",
    is_flag=True,
    help="Print the patpy-mcp version and exit.",
)
def run_app(
    transport: str,
    port: int,
    hostname: str,
    environment: EnvironmentType,
    show_version: bool,
) -> None:
    """Run the patpy-mcp server.

    The server speaks MCP over stdio by default; pass ``-t http`` or
    ``-t sse`` for network transports. ``MCP_TRANSPORT``, ``MCP_PORT``,
    ``MCP_HOSTNAME``, and ``MCP_ENVIRONMENT`` environment variables
    override the equivalent flags.
    """
    if show_version:
        from patpy_mcp import __version__

        click.echo(__version__)
        sys.exit(0)

    logging.basicConfig(
        level="DEBUG" if environment == EnvironmentType.DEVELOPMENT else "INFO",
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("patpy_mcp")

    from patpy_mcp.mcp import mcp

    transport_lower = transport.lower()
    logger.info("Starting patpy-mcp (env=%s, transport=%s)", environment.name, transport_lower)
    if transport_lower in {"http", "sse"}:
        mcp.run(transport=transport_lower, host=hostname, port=port)
    else:
        mcp.run(transport=transport_lower)


if __name__ == "__main__":
    run_app()
