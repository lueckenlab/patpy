"""Shared pytest fixtures for the patpy-mcp test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Force every test into a clean ``$PATPY_MCP_CACHE`` under ``tmp_path``.

    Marked ``autouse`` so contributors cannot accidentally write the
    real ``~/.cache/patpy-mcp/`` from a unit test.
    """
    monkeypatch.setenv("PATPY_MCP_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    yield
