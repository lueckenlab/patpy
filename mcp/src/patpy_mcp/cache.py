"""Local cache for datasets and indices fetched by the MCP server.

The cache layout is intentionally simple so other MCP servers (notably
``biocontext-ai/anndata-mcp``) can pick up downloaded files by absolute
path::

    ~/.cache/patpy-mcp/
        datasets/<source>/<dataset_id>/<asset_id>.h5ad
        datasets/<source>/<dataset_id>/<asset_id>.h5ad.meta.json
        index/<source>_<index_name>.json

Cache root resolution order:

1. ``$PATPY_MCP_CACHE`` (explicit override)
2. ``$XDG_CACHE_HOME/patpy-mcp`` (Linux convention)
3. ``~/.cache/patpy-mcp`` (fallback)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_SCHEMA_VERSION = 1


def cache_root() -> Path:
    """Return the on-disk cache root, creating it if needed."""
    if env := os.environ.get("PATPY_MCP_CACHE"):
        root = Path(env).expanduser().resolve()
    elif xdg := os.environ.get("XDG_CACHE_HOME"):
        root = Path(xdg).expanduser().resolve() / "patpy-mcp"
    else:
        root = Path.home() / ".cache" / "patpy-mcp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_dir(source: str, dataset_id: str) -> Path:
    """Return ``<cache_root>/datasets/<source>/<dataset_id>/`` (created)."""
    out = cache_root() / "datasets" / source / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def index_path(source: str, name: str) -> Path:
    """Return path for a JSON index file under ``<cache_root>/index/``."""
    out = cache_root() / "index"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{source}_{name}.json"


@dataclass(frozen=True)
class FileMetadata:
    """Sidecar describing a cached file. Serialised to ``<file>.meta.json``."""

    source: str
    dataset_id: str
    asset_id: str
    source_url: str
    size_bytes: int
    sha256: str
    fetched_at: float
    schema_version: int = CACHE_SCHEMA_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def write_metadata(target_file: Path, meta: FileMetadata) -> Path:
    """Write a sidecar ``.meta.json`` next to ``target_file``."""
    sidecar = target_file.with_suffix(target_file.suffix + ".meta.json")
    sidecar.write_text(json.dumps(meta.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def read_metadata(target_file: Path) -> FileMetadata | None:
    """Read the sidecar for ``target_file``, returning ``None`` if absent or stale."""
    sidecar = target_file.with_suffix(target_file.suffix + ".meta.json")
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    try:
        return FileMetadata(**payload)
    except TypeError:
        return None


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash a file to avoid loading it fully into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def now_timestamp() -> float:
    """Wrapper around :func:`time.time` for deterministic monkey-patching in tests."""
    return time.time()


def is_index_fresh(path: Path, ttl_seconds: float) -> bool:
    """Return ``True`` iff ``path`` exists and was modified within ``ttl_seconds``."""
    if not path.is_file():
        return False
    return (now_timestamp() - path.stat().st_mtime) < ttl_seconds
