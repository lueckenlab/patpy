"""Unit tests for the CellxGene Discover REST client.

We avoid pulling in `responses` or `pytest-httpx` as test dependencies by
hand-rolling a tiny ``FakeSession`` that satisfies the subset of the
``requests.Session`` API actually used by
:class:`patpy.mcp.sources.cellxgene.discover.DiscoverClient`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

pytest.importorskip("requests", reason="patpy[mcp] extras are required for these tests.")
import requests  # noqa: E402

from patpy.mcp.sources.cellxgene.discover import API_BASE, DiscoverClient  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Force the MCP cache into ``tmp_path`` for every test."""
    monkeypatch.setenv("PATPY_MCP_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    yield


def _fixture_collection() -> dict[str, Any]:
    return {
        "collection_id": "col-1",
        "id": "col-1",
        "name": "BRCA Atlas",
        "description": "Breast cancer atlas",
        "doi": "10.1000/x",
        "datasets": [],
    }


def _fixture_datasets() -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": "d-breast-1",
            "title": "Breast carcinoma single-cell atlas",
            "collection_id": "col-1",
            "collection_name": "BRCA Atlas",
            "organism": [{"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"}],
            "disease": [{"label": "breast carcinoma", "ontology_term_id": "MONDO:0007254"}],
            "tissue": [{"label": "breast", "ontology_term_id": "UBERON:0000310"}],
            "assay": [{"label": "10x 3' v3", "ontology_term_id": "EFO:0009922"}],
            "cell_count": 200_000,
            "donor_count": 50,
            "primary_data": True,
            "assets": [
                {
                    "dataset_asset_id": "asset-h5ad",
                    "filetype": "H5AD",
                    "filename": "atlas.h5ad",
                    "filesize": 1024,
                    "url": "https://files.example/asset-h5ad",
                }
            ],
        },
        {
            "dataset_id": "d-lung-1",
            "title": "Lung adenocarcinoma cohort",
            "collection_id": "col-2",
            "collection_name": "Lung Cohort",
            "organism": [{"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"}],
            "disease": [{"label": "lung adenocarcinoma", "ontology_term_id": "MONDO:0005061"}],
            "tissue": [{"label": "lung", "ontology_term_id": "UBERON:0002048"}],
            "assay": [{"label": "Smart-seq2", "ontology_term_id": "EFO:0008931"}],
            "cell_count": 30_000,
            "donor_count": 12,
            "primary_data": True,
            "assets": [],
        },
    ]


class _FakeResponse:
    """Mimics the slice of ``requests.Response`` we touch."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any | None = None,
        content: bytes = b"",
        chunk_size: int = 256,
    ) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json = json_data
        self._content = content
        self._chunk_size = chunk_size
        self.text = ""

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)  # type: ignore[arg-type]

    def iter_content(self, chunk_size: int = 4096):
        size = chunk_size or self._chunk_size or 256
        for i in range(0, len(self._content), size):
            yield self._content[i : i + size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    """Records calls and returns canned responses keyed by URL path."""

    def __init__(self, json_routes: dict[str, Any], download_routes: dict[str, bytes]) -> None:
        self.json_routes = json_routes
        self.download_routes = download_routes
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict | None, bool]] = []

    def get(
        self,
        url: str,
        params: dict | None = None,
        timeout: tuple[float, float] | None = None,
        stream: bool = False,
    ) -> _FakeResponse:
        path = urlsplit(url).path.rstrip("/").lstrip("/")
        self.calls.append((url, path, params, stream))

        if stream:
            content = self.download_routes.get(url)
            if content is None:
                return _FakeResponse(status_code=404)
            return _FakeResponse(content=content)

        # Normalise the API path so test fixtures key on suffix only.
        if url.startswith(API_BASE):
            path = url[len(API_BASE) :].strip("/")
        if path in self.json_routes:
            return _FakeResponse(json_data=self.json_routes[path])
        return _FakeResponse(status_code=404, json_data={"error": f"No route for {path}"})


@pytest.fixture
def fake_session():
    payload_bytes = b"hello world\n" * 64
    routes = {
        "datasets": _fixture_datasets(),
        "collections": [_fixture_collection()],
        "collections/col-1": _fixture_collection(),
        "datasets/d-breast-1": _fixture_datasets()[0],
    }
    downloads = {"https://files.example/asset-h5ad": payload_bytes}
    return _FakeSession(json_routes=routes, download_routes=downloads), payload_bytes


@pytest.fixture
def client(fake_session):
    session, _ = fake_session
    return DiscoverClient(session=session, max_retries=1)


def test_search_filters_by_disease_label(client, fake_session):
    results = client.search_datasets(disease=["breast carcinoma"])
    assert len(results) == 1
    assert results[0]["dataset_id"] == "d-breast-1"
    assert results[0]["explorer_url"].endswith("/collections/col-1/datasets/d-breast-1")


def test_search_filters_by_disease_ontology_id(client):
    results = client.search_datasets(disease=["MONDO:0007254"])
    assert [r["dataset_id"] for r in results] == ["d-breast-1"]


def test_search_filters_by_min_cells_and_organism(client):
    none_match = client.search_datasets(min_cells=500_000)
    assert none_match == []
    all_match = client.search_datasets(organism="Homo sapiens", min_cells=10_000)
    assert {r["dataset_id"] for r in all_match} == {"d-breast-1", "d-lung-1"}


def test_search_supports_offset_and_limit(client):
    page = client.search_datasets(limit=1)
    assert len(page) == 1
    page2 = client.search_datasets(limit=1, offset=1)
    assert len(page2) == 1
    assert page[0]["dataset_id"] != page2[0]["dataset_id"]


def test_list_datasets_response_is_cached_on_disk(client, fake_session):
    session, _ = fake_session
    client.search_datasets()
    n_after_first = sum(1 for _, p, _, _ in session.calls if p.endswith("datasets"))
    client.search_datasets()
    n_after_second = sum(1 for _, p, _, _ in session.calls if p.endswith("datasets"))
    assert n_after_first == n_after_second == 1, (
        "The /datasets list endpoint should only be hit once thanks to the on-disk index cache."
    )


def test_list_disease_terms_includes_label_and_ontology_id(client):
    terms = client.list_disease_terms(prefix="breast")
    assert any(t["label"] == "breast carcinoma" and t["ontology_term_id"] == "MONDO:0007254" for t in terms)


def test_get_dataset_returns_assets_block(client):
    meta = client.get_dataset("d-breast-1")
    assert meta["dataset_id"] == "d-breast-1"
    assert meta["assets"] and meta["assets"][0]["filetype"] == "H5AD"


def test_download_writes_file_with_correct_sha256(client, fake_session, tmp_path):
    _, payload_bytes = fake_session
    expected_sha = hashlib.sha256(payload_bytes).hexdigest()

    result = client.download_dataset("d-breast-1", out_dir=str(tmp_path / "out"))

    assert result["cached"] is False
    assert result["sha256"] == expected_sha
    assert result["size_bytes"] == len(payload_bytes)
    assert Path(result["local_path"]).read_bytes() == payload_bytes
    sidecar = json.loads(Path(result["local_path"] + ".meta.json").read_text())
    assert sidecar["sha256"] == expected_sha
    assert sidecar["source_url"] == "https://files.example/asset-h5ad"


def test_download_is_idempotent_and_reports_cached(client, fake_session, tmp_path):
    out = str(tmp_path / "out")
    first = client.download_dataset("d-breast-1", out_dir=out)
    second = client.download_dataset("d-breast-1", out_dir=out)
    assert first["sha256"] == second["sha256"]
    assert first["cached"] is False
    assert second["cached"] is True


def test_download_refuses_above_max_size(client):
    with pytest.raises(ValueError, match="exceeds max_size_gb"):
        client.download_dataset("d-breast-1", max_size_gb=0.0000001)


def test_download_unknown_format_raises(client):
    with pytest.raises(ValueError, match="No RDS asset"):
        client.download_dataset("d-breast-1", asset_format="RDS")
