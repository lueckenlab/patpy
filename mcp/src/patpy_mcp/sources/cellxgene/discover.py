"""CellxGene Discover Curation REST API client.

Wraps the public, unauthenticated read-only endpoints under
``https://api.cellxgene.cziscience.com/curation/v1/`` with ergonomic
helpers used by the patpy MCP tools:

* :meth:`DiscoverClient.search_datasets` -- list + client-side filter
* :meth:`DiscoverClient.get_dataset` / :meth:`get_collection`
* :meth:`DiscoverClient.list_collections`
* :meth:`DiscoverClient.list_disease_terms` / :meth:`list_tissue_terms`
* :meth:`DiscoverClient.download_dataset` -- streaming, SHA-256 verified

Server-side filters on the Curation API are limited so most filtering is
applied client-side after fetching the (paginated) full dataset list.
The full list is small enough to cache for 24 h, which we do via
:func:`patpy_mcp.cache.index_path`.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from patpy_mcp import cache

if TYPE_CHECKING:
    import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.cellxgene.cziscience.com/curation/v1"
EXPLORER_BASE = "https://cellxgene.cziscience.com"
INDEX_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TIMEOUT = (10, 60)
"""Connect / read timeouts for normal JSON requests, in seconds."""


def _require_requests():
    """Lazy import so :mod:`patpy_mcp` can be imported without ``requests`` installed."""
    try:
        import requests
    except ImportError as err:  # pragma: no cover  (covered via runtime install check)
        raise ImportError(
            "The 'requests' package is required for the CellxGene MCP source. "
            "Install patpy-mcp with: pip install patpy-mcp"
        ) from err
    return requests


@dataclass
class DownloadStreamResult:
    """Internal record produced by :func:`_stream_to_file`."""

    size_bytes: int
    sha256: str


class DiscoverClient:
    """Thin, retrying client for the CellxGene Discover Curation API."""

    def __init__(
        self,
        base_url: str = API_BASE,
        session: requests.Session | None = None,
        max_retries: int = 3,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            requests = _require_requests()
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Accept": "application/json",
                    "User-Agent": "patpy-mcp (+https://github.com/lueckenlab/patpy)",
                }
            )
        return self._session

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``{base_url}/{path}`` returning parsed JSON, with simple retry."""
        requests = _require_requests()
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as err:
                last_err = err
                logger.warning("Network error fetching %s (attempt %d): %s", url, attempt + 1, err)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_err = RuntimeError(
                    f"CellxGene API returned {response.status_code} for {url}: {response.text[:200]}"
                )
                logger.warning("Retrying %s after status %d", url, response.status_code)
                continue
            if not response.ok:
                response.raise_for_status()
            return response.json()
        raise RuntimeError(f"CellxGene API request failed after {self.max_retries} attempts: {last_err}")

    def list_collections_raw(self) -> list[dict[str, Any]]:
        """Return the raw ``GET /collections`` payload (cached on disk for 24 h)."""
        cached = cache.index_path("cellxgene", "collections")
        if cache.is_index_fresh(cached, INDEX_TTL_SECONDS):
            try:
                return json.loads(cached.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        payload = self._get_json("collections")
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected /collections response shape (expected a list).")
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def list_datasets_raw(self) -> list[dict[str, Any]]:
        """Return the raw ``GET /datasets`` payload (cached on disk for 24 h)."""
        cached = cache.index_path("cellxgene", "datasets")
        if cache.is_index_fresh(cached, INDEX_TTL_SECONDS):
            try:
                return json.loads(cached.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        payload = self._get_json("datasets")
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected /datasets response shape (expected a list).")
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def get_collection_raw(self, collection_id: str) -> dict[str, Any]:
        """Return the raw ``GET /collections/{id}`` payload."""
        return self._get_json(f"collections/{collection_id}")

    def _resolve_collection_id(self, dataset_id: str) -> str:
        """Find a dataset's parent collection by scanning the cached dataset list.

        The CellxGene Curation API only exposes per-dataset metadata
        under ``/collections/{cid}/datasets/{dsid}`` -- there is no
        flat ``/datasets/{dsid}`` endpoint. Since :meth:`list_datasets_raw`
        already caches the full dataset directory for 24 h, looking up
        the collection id is a cheap dict scan in the common case.
        """
        for d in self.list_datasets_raw():
            if (d.get("dataset_id") or d.get("id")) == dataset_id:
                cid = d.get("collection_id")
                if cid:
                    return cid
                break
        raise ValueError(
            f"Could not locate collection for dataset {dataset_id!r}. "
            "It may have been removed from CellxGene Discover or be "
            "private; pass an explicit collection_id via "
            "DiscoverClient.get_collection_raw if you have one."
        )

    def get_dataset_raw(self, dataset_id: str, collection_id: str | None = None) -> dict[str, Any]:
        """Return the raw per-dataset payload, including assets.

        Hits ``GET /curation/v1/collections/{collection_id}/datasets/{dataset_id}``.
        ``collection_id`` is resolved from the cached dataset list when not
        supplied explicitly.
        """
        if collection_id is None:
            collection_id = self._resolve_collection_id(dataset_id)
        return self._get_json(f"collections/{collection_id}/datasets/{dataset_id}")

    def list_collections(self, query: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        """List collections, optionally filtered by free-text substring on name/description."""
        items = self.list_collections_raw()
        if query:
            q = query.lower()
            items = [
                c for c in items if q in (c.get("name") or "").lower() or q in (c.get("description") or "").lower()
            ]
        return [_collection_summary(c) for c in items[:limit]]

    def get_collection(self, collection_id: str) -> dict[str, Any]:
        """Return a normalised collection record including its dataset summaries."""
        return _collection_summary(self.get_collection_raw(collection_id), include_datasets=True)

    def search_datasets(
        self,
        query: str | None = None,
        disease: list[str] | None = None,
        tissue: list[str] | None = None,
        organism: str | None = "Homo sapiens",
        assay: list[str] | None = None,
        min_cells: int | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Client-side-filtered dataset search."""
        items = self.list_datasets_raw()

        if query:
            q = query.lower()
            items = [d for d in items if q in (d.get("title") or "").lower()]

        if organism:
            items = [d for d in items if _matches_term_set(d.get("organism"), [organism])]
        if disease:
            items = [d for d in items if _matches_term_set(d.get("disease"), disease)]
        if tissue:
            items = [d for d in items if _matches_term_set(d.get("tissue"), tissue)]
        if assay:
            items = [d for d in items if _matches_term_set(d.get("assay"), assay)]
        if min_cells is not None:
            items = [d for d in items if (d.get("cell_count") or 0) >= min_cells]

        items = items[offset : offset + limit]
        return [_dataset_summary(d) for d in items]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Return a normalised dataset record including assets and explorer URL."""
        return _dataset_metadata(self.get_dataset_raw(dataset_id))

    def list_disease_terms(self, prefix: str | None = None, limit: int = 200) -> list[dict[str, str]]:
        """Distinct disease terms across all datasets."""
        return _collect_terms(self.list_datasets_raw(), key="disease", prefix=prefix, limit=limit)

    def list_tissue_terms(self, prefix: str | None = None, limit: int = 200) -> list[dict[str, str]]:
        """Distinct tissue terms across all datasets."""
        return _collect_terms(self.list_datasets_raw(), key="tissue", prefix=prefix, limit=limit)

    def download_dataset(
        self,
        dataset_id: str,
        asset_format: str = "H5AD",
        out_dir: str | None = None,
        max_size_gb: float = 10.0,
        force: bool = False,
    ) -> dict[str, Any]:
        """Stream-download a dataset asset to the local cache.

        Returns a dict with ``local_path``, ``size_bytes``, ``sha256``,
        ``cached``, and ``source_url`` keys.
        """
        meta = self.get_dataset_raw(dataset_id)
        asset = _pick_asset(meta.get("assets") or [], asset_format)
        if asset is None:
            available = sorted({(a.get("filetype") or "?") for a in meta.get("assets") or []})
            raise ValueError(
                f"No {asset_format} asset found for dataset {dataset_id}. Available formats: {available or 'none'}"
            )

        asset_id = asset.get("filename") or asset.get("dataset_asset_id") or asset_format.lower()
        url = asset.get("url") or asset.get("presigned_url")
        if not url:
            raise RuntimeError(f"Asset {asset_id} for dataset {dataset_id} has no download URL.")

        size_estimate = int(asset.get("filesize") or 0)
        max_size_bytes = int(max_size_gb * 1024**3)
        if size_estimate and size_estimate > max_size_bytes:
            raise ValueError(
                f"Dataset {dataset_id} asset is {size_estimate / 1024**3:.1f} GB which exceeds "
                f"max_size_gb={max_size_gb}. Pass a higher max_size_gb to override."
            )

        target_dir = Path(out_dir).expanduser().resolve() if out_dir else cache.dataset_dir("cellxgene", dataset_id)
        target_file = target_dir / Path(asset_id).name

        existing = cache.read_metadata(target_file)
        if not force and target_file.is_file() and existing is not None:
            return {
                "source": "cellxgene",
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "local_path": str(target_file),
                "size_bytes": existing.size_bytes,
                "sha256": existing.sha256,
                "cached": True,
                "source_url": existing.source_url,
            }

        stream_result = _stream_to_file(self.session, url, target_file, max_size_bytes, self.timeout)
        meta_record = cache.FileMetadata(
            source="cellxgene",
            dataset_id=dataset_id,
            asset_id=asset_id,
            source_url=url,
            size_bytes=stream_result.size_bytes,
            sha256=stream_result.sha256,
            fetched_at=cache.now_timestamp(),
        )
        cache.write_metadata(target_file, meta_record)

        return {
            "source": "cellxgene",
            "dataset_id": dataset_id,
            "asset_id": asset_id,
            "local_path": str(target_file),
            "size_bytes": stream_result.size_bytes,
            "sha256": stream_result.sha256,
            "cached": False,
            "source_url": url,
        }


def _stream_to_file(
    session: requests.Session,
    url: str,
    target: Path,
    max_size_bytes: int,
    timeout: tuple[float, float],
) -> DownloadStreamResult:
    """Stream ``url`` to ``target`` while computing SHA-256 and enforcing size limit.

    Writes to a temp file first and atomically replaces ``target`` on
    success so a partial download never poisons the cache.
    """
    _require_requests()
    with session.get(url, stream=True, timeout=timeout) as response:
        if not response.ok:
            response.raise_for_status()

        import hashlib

        h = hashlib.sha256()
        size = 0

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd = tempfile.NamedTemporaryFile(
            prefix=target.name + ".",
            suffix=".part",
            dir=str(target.parent),
            delete=False,
        )
        tmp_path = Path(tmp_fd.name)
        try:
            with tmp_fd:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if max_size_bytes and size > max_size_bytes:
                        raise ValueError(f"Download exceeded max_size_bytes={max_size_bytes} while streaming {url}")
                    tmp_fd.write(chunk)
                    h.update(chunk)
            shutil.move(str(tmp_path), str(target))
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    return DownloadStreamResult(size_bytes=size, sha256=h.hexdigest())


def _term_entry(value: Any) -> dict[str, str] | None:
    """Normalise ontology entries that come in as ``{"label", "ontology_term_id"}`` or strings."""
    if isinstance(value, dict):
        label = value.get("label") or value.get("name")
        term_id = value.get("ontology_term_id") or value.get("id")
        if not label and not term_id:
            return None
        return {"label": label or term_id, "ontology_term_id": term_id or ""}
    if isinstance(value, str) and value:
        return {"label": value, "ontology_term_id": ""}
    return None


def _term_labels(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        entry = _term_entry(v)
        if entry is not None:
            out.append(entry["label"])
    return out


def _matches_term_set(values: Any, queries: list[str]) -> bool:
    """Return ``True`` if any of ``queries`` (label or ontology id) is in ``values``."""
    needles = [q.lower() for q in queries if q]
    if not needles:
        return True
    for value in values or []:
        entry = _term_entry(value)
        if entry is None:
            continue
        haystack = (entry.get("label") or "").lower(), (entry.get("ontology_term_id") or "").lower()
        for needle in needles:
            if any(needle == h or needle in h for h in haystack if h):
                return True
    return False


def _collect_terms(
    datasets: list[dict[str, Any]],
    key: str,
    prefix: str | None,
    limit: int,
) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for d in datasets:
        for raw in d.get(key) or []:
            entry = _term_entry(raw)
            if entry is None:
                continue
            term_id = entry["ontology_term_id"] or entry["label"]
            seen.setdefault(term_id, entry)
    items = list(seen.values())
    if prefix:
        p = prefix.lower()
        items = [e for e in items if e["label"].lower().startswith(p) or e["ontology_term_id"].lower().startswith(p)]
    items.sort(key=lambda e: e["label"].lower())
    return items[:limit]


def _dataset_summary(d: dict[str, Any]) -> dict[str, Any]:
    dataset_id = d.get("dataset_id") or d.get("id") or ""
    collection_id = d.get("collection_id") or ""
    return {
        "source": "cellxgene",
        "dataset_id": dataset_id,
        "title": d.get("title") or "",
        "collection_id": collection_id,
        "collection_name": d.get("collection_name") or "",
        "organism": _term_labels(d.get("organism") or []),
        "disease": _term_labels(d.get("disease") or []),
        "tissue": _term_labels(d.get("tissue") or []),
        "assay": _term_labels(d.get("assay") or []),
        "cell_count": int(d.get("cell_count") or 0),
        "donor_count": int(d.get("donor_count") or 0),
        "primary_data": bool(d.get("primary_data")) if d.get("primary_data") is not None else True,
        "explorer_url": _explorer_url(collection_id, dataset_id),
    }


def _dataset_metadata(d: dict[str, Any]) -> dict[str, Any]:
    summary = _dataset_summary(d)
    summary.update(
        {
            "description": d.get("description") or "",
            "schema_version": d.get("schema_version") or "",
            "assets": [
                {
                    "asset_id": a.get("dataset_asset_id") or a.get("filename") or "",
                    "filetype": a.get("filetype") or "",
                    "filename": a.get("filename") or "",
                    "filesize": a.get("filesize") or 0,
                    "url": a.get("url") or a.get("presigned_url") or "",
                }
                for a in d.get("assets") or []
            ],
            "raw": d,
        }
    )
    return summary


def _collection_summary(c: dict[str, Any], include_datasets: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": "cellxgene",
        "collection_id": c.get("collection_id") or c.get("id") or "",
        "name": c.get("name") or "",
        "description": (c.get("description") or "")[:500],
        "doi": c.get("doi") or "",
        "publisher_metadata": c.get("publisher_metadata") or {},
        "dataset_count": len(c.get("datasets") or []),
    }
    if include_datasets:
        out["datasets"] = [_dataset_summary(d) for d in c.get("datasets") or []]
    return out


def _pick_asset(assets: list[dict[str, Any]], filetype: str) -> dict[str, Any] | None:
    target = filetype.lower()
    for a in assets:
        if (a.get("filetype") or "").lower() == target:
            return a
    return None


def _explorer_url(collection_id: str, dataset_id: str) -> str:
    if not collection_id or not dataset_id:
        return ""
    return f"{EXPLORER_BASE}/collections/{collection_id}/datasets/{dataset_id}"
