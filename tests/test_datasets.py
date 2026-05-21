import os
from pathlib import Path

import pytest
from anndata import AnnData

import patpy
from patpy.datasets import DatasetInfo

CACHE_DIR = Path(os.environ["PATPY_TEST_DATASETDIR"]) if os.environ.get("PATPY_TEST_DATASETDIR") else None


def _datasetdir(tmp_path: Path) -> Path:
    """Return a persistent cache dir when available, otherwise fall back to tmp_path."""
    if CACHE_DIR is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR
    return tmp_path


def _check_dataset_info(adata: AnnData, info: DatasetInfo) -> None:
    """Common assertions tying a returned DatasetInfo back to the AnnData."""
    assert isinstance(info, DatasetInfo)
    assert info.n_cells == adata.n_obs
    assert info.n_features == adata.n_vars
    assert info.sample_key in adata.obs.columns
    assert adata.obs[info.sample_key].nunique() == info.n_samples
    assert info.cell_type_key in adata.obs.columns
    for col in info.sample_metadata_columns:
        assert col in adata.obs.columns


@pytest.mark.dataset
def test_combat(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.combat()

        assert isinstance(adata, AnnData)
        assert adata.n_obs == 783677
        assert adata.n_vars == 3000
        assert "PCs" in adata.varm

        adata, info = patpy.datasets.combat(return_dataset_info=True)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
def test_combat_load_metadata(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata, meta_adata = patpy.datasets.combat(load_metadata=True)
        assert isinstance(adata, AnnData)
        assert isinstance(meta_adata, AnnData)
        # Sample-level metadata: one row per COMBAT sample.
        assert meta_adata.n_obs == 137

        adata, meta_adata, info = patpy.datasets.combat(load_metadata=True, return_dataset_info=True)
        assert isinstance(meta_adata, AnnData)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
def test_hlca(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.hlca()

        assert isinstance(adata, AnnData)
        assert adata.n_obs == 1687127
        assert adata.n_vars == 3000
        assert "PCs" in adata.varm

        adata, info = patpy.datasets.hlca(return_dataset_info=True)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
def test_onek1k(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.onek1k()

        assert isinstance(adata, AnnData)
        assert adata.n_obs == 1248980
        assert adata.n_vars == 3000
        assert "PCs" in adata.varm

        adata, info = patpy.datasets.onek1k(return_dataset_info=True)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
def test_stephenson(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.stephenson()

        assert isinstance(adata, AnnData)
        assert adata.n_obs == 639482
        assert adata.n_vars == 3000
        assert "PCs" in adata.varm

        adata, info = patpy.datasets.stephenson(return_dataset_info=True)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
def test_ticatlas(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.ticatlas()

        assert isinstance(adata, AnnData)
        assert adata.n_obs == 267547
        assert adata.n_vars == 3000
        assert "PCs" in adata.varm

        adata, info = patpy.datasets.ticatlas(return_dataset_info=True)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
def test_combat_stephenson(tmp_path):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.combat_stephenson()

        assert isinstance(adata, AnnData)
        assert adata.n_obs == 1399435
        assert adata.n_vars == 1856
        assert "X_raw_counts" in adata.layers

        adata, info = patpy.datasets.combat_stephenson(return_dataset_info=True)
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


@pytest.mark.dataset
@pytest.mark.parametrize(
    ("split", "n_obs", "n_samples"),
    [
        ("main", 4918140, 817),
        ("external", 572872, 86),
        ("validation", 849922, 144),
    ],
)
def test_inflammation_atlas(tmp_path, split, n_obs, n_samples):
    original = patpy.settings.datasetdir
    patpy.settings.datasetdir = _datasetdir(tmp_path)
    try:
        adata = patpy.datasets.inflammation_atlas(split=split)

        assert isinstance(adata, AnnData)
        assert adata.n_obs == n_obs
        assert adata.n_vars == 0
        assert "X_scANVI_atlas" in adata.obsm

        adata, info = patpy.datasets.inflammation_atlas(split=split, return_dataset_info=True)
        assert info.n_samples == n_samples
        _check_dataset_info(adata, info)
    finally:
        patpy.settings.datasetdir = original


def test_inflammation_atlas_invalid_split():
    with pytest.raises(ValueError):
        patpy.datasets.inflammation_atlas(split="banana")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "loader",
    [
        patpy.datasets.combat,
        patpy.datasets.hlca,
        patpy.datasets.onek1k,
        patpy.datasets.stephenson,
        patpy.datasets.ticatlas,
        patpy.datasets.combat_stephenson,
        patpy.datasets.inflammation_atlas,
    ],
)
def test_kind_raw_not_implemented(loader):
    """Raw versions of all datasets should raise NotImplementedError until URLs are wired up."""
    with pytest.raises(NotImplementedError):
        loader(kind="raw")


def test_kind_invalid_value():
    """An unknown kind should raise a clear ValueError."""
    with pytest.raises(ValueError):
        patpy.datasets.combat(kind="banana")  # type: ignore[arg-type]
