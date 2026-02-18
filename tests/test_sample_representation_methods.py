import numpy as np
import pytest

from patpy.tl.sample_representation import (
    CellGroupComposition,
    DiffusionEarthMoverDistance,
    GloScope,
    GloScope_py,
    GroupedPseudobulk,
    MOFA,
    MrVI,
    PhEMD,
    PILOT,
    Pseudobulk,
    RandomVector,
    SCPoli,
    WassersteinTSNE,
    calculate_average_without_nans,
    valid_aggregate,
    valid_distance_metric,
)

SAMPLE_KEY = "sample_id"
CELL_KEY = "cell_type"
# pbmc3k_processed uses louvain cluster labels as cell-type annotations
PBMC_CELL_KEY = "louvain"

LIGHTWEIGHT_METHODS = [
    (Pseudobulk, {"layer": "X"}),
    (GroupedPseudobulk, {"layer": "X"}),
    (RandomVector, {}),
    (CellGroupComposition, {}),
]


def _assert_distances(distances, n_samples, uns, uns_key, *, symmetric=True):
    """Assert that distances is a valid (n_samples, n_samples) matrix stored in uns."""
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (n_samples, n_samples)
    if symmetric:
        assert np.allclose(distances, distances.T, atol=1e-5)
    assert uns_key in uns
    assert np.array_equal(uns[uns_key], distances)


def _assert_cache_respected(method, uns, computed_distances, extra_uns=None):
    """Assert that calculate_distance_matrix() returns the cached value without recomputing."""
    sentinel = np.full_like(computed_distances, fill_value=-1.0)
    uns[method.DISTANCES_UNS_KEY] = sentinel  # Rewrite the calculated matrix with an arbitrary one
    if extra_uns:  # Needed for WassersteinTSNE as it stores extra info in uns
        uns.update(extra_uns)
    assert np.array_equal(method.calculate_distance_matrix(), sentinel)


# ---------------------------------------------------------------------------
# Lightweight methods
# ---------------------------------------------------------------------------


# Verifies that every lightweight sample representation computes and stores a symmetric distance matrix.
@pytest.mark.parametrize("method_cls, kwargs", LIGHTWEIGHT_METHODS)
def test_distance_matrix_is_computed_and_cached(method_cls, kwargs, synthetic_adata):
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = method_cls(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, **kwargs)
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    assert isinstance(distances, np.ndarray)
    assert distances.shape == (n_samples, n_samples)
    assert np.allclose(distances, distances.T)
    assert method.DISTANCES_UNS_KEY in adata.uns
    assert np.array_equal(adata.uns[method.DISTANCES_UNS_KEY], distances)


# Ensures all methods respect cached distances unless explicitly forced to recompute.
@pytest.mark.parametrize("method_cls, kwargs", LIGHTWEIGHT_METHODS)
def test_distance_matrix_uses_cache_when_present(method_cls, kwargs, synthetic_adata):
    adata = synthetic_adata.copy()

    method = method_cls(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, **kwargs)
    method.prepare_anndata(adata)
    baseline = method.calculate_distance_matrix(force=True)

    cached = np.full_like(baseline, fill_value=-1.0)
    adata.uns[method.DISTANCES_UNS_KEY] = cached

    distances = method.calculate_distance_matrix()

    assert np.array_equal(distances, cached)
    assert adata.uns[method.DISTANCES_UNS_KEY] is cached


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


# Validates averaging logic with and without NaNs, including default fill behavior.
def test_calculate_average_without_nans(integer_matrix, float_matrix_with_nans, nan_heavy_matrix):
    averages, sample_sizes = calculate_average_without_nans(integer_matrix, axis=0)
    assert np.allclose(averages, [4, 5, 6])
    assert np.array_equal(sample_sizes, [3, 3, 3])

    averages, sample_sizes = calculate_average_without_nans(float_matrix_with_nans, axis=0)
    assert np.allclose(averages, [2.5, 5, 7.5])
    assert np.array_equal(sample_sizes, [2, 2, 2])

    averages, sample_sizes = calculate_average_without_nans(nan_heavy_matrix, axis=0, default_value=0)
    assert np.allclose(averages, [0, 5, 6])
    assert np.array_equal(sample_sizes, [0, 2, 1])

    averages = calculate_average_without_nans(nan_heavy_matrix, axis=0, return_sample_sizes=False)
    assert np.allclose(averages, [0, 5, 6])


def test_valid_aggregate_raises_for_unknown_function():
    with pytest.raises(ValueError, match="not supported"):
        valid_aggregate("geometric_mean")


def test_valid_aggregate_returns_callable_for_known_functions():
    for name in ("mean", "median", "sum"):
        fn = valid_aggregate(name)
        assert callable(fn)


def test_valid_distance_metric_raises_for_unknown_metric():
    with pytest.raises(ValueError, match="not supported"):
        valid_distance_metric("hamming")


def test_valid_distance_metric_returns_name_for_known_metrics():
    for metric in ("euclidean", "cosine", "cityblock"):
        assert valid_distance_metric(metric) == metric


# ---------------------------------------------------------------------------
# MrVI (requires scvi-tools; needs raw count data → synthetic_adata)
# ---------------------------------------------------------------------------


def test_mrvi(synthetic_adata):
    pytest.importorskip("scvi")
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = MrVI(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, max_epochs=1)
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, adata.uns, distances)


# ---------------------------------------------------------------------------
# WassersteinTSNE (requires WassersteinTSNE package)
# ---------------------------------------------------------------------------


def test_wasserstein_tsne(pbmc3k_adata):
    pytest.importorskip("WassersteinTSNE")
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = WassersteinTSNE(
        sample_key=SAMPLE_KEY,
        cell_group_key=PBMC_CELL_KEY,
        replicate_key=PBMC_CELL_KEY,
        layer="X_pca",
    )
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix()

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY, symmetric=False)
    _assert_cache_respected(method, adata.uns, distances, extra_uns={"wasserstein_covariance_weight": 0.5})


# ---------------------------------------------------------------------------
# PILOT (requires pilotpy)
# ---------------------------------------------------------------------------


def test_pilot(pbmc3k_adata):
    pytest.importorskip("pilotpy", exc_type=Exception)
    adata = pbmc3k_adata.copy()
    adata.obs["state"] = "control"
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = PILOT(
        sample_key=SAMPLE_KEY,
        cell_group_key=PBMC_CELL_KEY,
        sample_state_col="state",
        layer="X_pca",
    )
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, adata.uns, distances)


# ---------------------------------------------------------------------------
# SCPoli (requires scarches; needs raw count data → synthetic_adata)
# ---------------------------------------------------------------------------


def test_scpoli(synthetic_adata):
    pytest.importorskip("scarches")
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = SCPoli(
        sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, n_epochs=1, pretraining_epochs=1
    )
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    # SCPoli replaces self.adata with an optimized copy; check through the method object
    _assert_distances(distances, n_samples, method.adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, method.adata.uns, distances)


# ---------------------------------------------------------------------------
# PhEMD (requires ot and phate)
# ---------------------------------------------------------------------------


def test_phemd(pbmc3k_adata):
    pytest.importorskip("ot")
    pytest.importorskip("phate", exc_type=ImportError)
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = PhEMD(sample_key=SAMPLE_KEY, cell_group_key=PBMC_CELL_KEY, n_clusters=3)
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, adata.uns, distances)


# ---------------------------------------------------------------------------
# DiffusionEarthMoverDistance (requires DiffusionEMD)
# ---------------------------------------------------------------------------


def test_diffusion_emd(pbmc3k_adata):
    pytest.importorskip("DiffusionEMD")
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = DiffusionEarthMoverDistance(
        sample_key=SAMPLE_KEY, cell_group_key=PBMC_CELL_KEY, layer="X_pca"
    )
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, adata.uns, distances)


# ---------------------------------------------------------------------------
# MOFA (requires mofapy2)
# ---------------------------------------------------------------------------


def test_mofa(pbmc3k_adata):
    pytest.importorskip("mofapy2")
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = MOFA(
        sample_key=SAMPLE_KEY,
        cell_group_key=PBMC_CELL_KEY,
        n_factors=3,
        iterations=10,
        quiet=True,
    )
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, adata.uns, distances)


# ---------------------------------------------------------------------------
# GloScope R-based (requires rpy2 and GloScope R package)
# ---------------------------------------------------------------------------


def _skip_if_gloscope_r_unavailable():
    """Skip the test if rpy2 or the GloScope R package are not available."""
    pytest.importorskip("rpy2")
    try:
        import rpy2.robjects as ro  # noqa: PLC0415

        ro.r("library(GloScope)")
    except Exception as exc:
        pytest.skip(f"GloScope R package not available: {exc}")


def test_gloscope_r(pbmc3k_adata):
    _skip_if_gloscope_r_unavailable()
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = GloScope(sample_key=SAMPLE_KEY, cell_group_key=PBMC_CELL_KEY, layer="X_pca")
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY, symmetric=False)
    _assert_cache_respected(method, adata.uns, distances)


# ---------------------------------------------------------------------------
# GloScope Python CPU (requires pynndescent)
# ---------------------------------------------------------------------------


def test_gloscope_py(pbmc3k_adata):
    pytest.importorskip("pynndescent")
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = GloScope_py(sample_key=SAMPLE_KEY, cell_group_key=PBMC_CELL_KEY, layer="X_pca")
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    _assert_distances(distances, n_samples, adata.uns, method.DISTANCES_UNS_KEY)
    _assert_cache_respected(method, adata.uns, distances)
