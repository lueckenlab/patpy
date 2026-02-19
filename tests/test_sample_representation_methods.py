import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from patpy.tl.sample_representation import (
    MOFA,
    PILOT,
    CellGroupComposition,
    DiffusionEarthMoverDistance,
    GloScope,
    GloScope_py,
    GroupedPseudobulk,
    MrVI,
    PhEMD,
    Pseudobulk,
    RandomVector,
    SCPoli,
    WassersteinTSNE,
    _remove_negative_distances,
    calculate_average_without_nans,
    correlate_cell_type_expression,
    correlate_composition,
    create_colormap,
    make_matrix_symmetric,
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

    method = SCPoli(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, n_epochs=1, pretraining_epochs=1)
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

    method = DiffusionEarthMoverDistance(sample_key=SAMPLE_KEY, cell_group_key=PBMC_CELL_KEY, layer="X_pca")
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
        import rpy2.robjects as ro

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


# ---------------------------------------------------------------------------
# make_matrix_symmetric
# ---------------------------------------------------------------------------


def test_make_matrix_symmetric_leaves_symmetric_matrix_unchanged():
    m = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    result = make_matrix_symmetric(m)
    assert np.array_equal(result, m)


def test_make_matrix_symmetric_fixes_asymmetric_matrix():
    m = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning, match="not symmetric"):
        result = make_matrix_symmetric(m)
    assert np.allclose(result, result.T)
    assert np.isclose(result[0, 1], 1.0)  # (2 + 0) / 2
    assert np.isclose(result[1, 2], 2.0)  # (4 + 0) / 2


# ---------------------------------------------------------------------------
# _remove_negative_distances
# ---------------------------------------------------------------------------


def test_remove_negative_distances_leaves_non_negative_unchanged():
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = _remove_negative_distances(d)
    assert np.array_equal(result, d)


def test_remove_negative_distances_clips_to_zero_and_warns():
    d = np.array([[0.0, -0.001, 1.0], [-0.001, 0.0, 2.0], [1.0, 2.0, 0.0]])
    with pytest.warns(UserWarning, match="negative"):
        result = _remove_negative_distances(d)
    assert (result >= 0).all()
    assert result[0, 1] == 0.0


# ---------------------------------------------------------------------------
# GloScope_py.kl_divergence (static math method)
# ---------------------------------------------------------------------------


def test_kl_divergence_equal_distances():
    # When r_i == r_j the log ratio is 0, so KL = log(m_j / (m_i - 1))
    r = np.ones(10)
    result = GloScope_py.kl_divergence(r_i=r, r_j=r, m_i=10, m_j=10, d=5)
    assert np.isclose(result, np.log(10 / 9))


def test_kl_divergence_returns_scalar():
    rng = np.random.default_rng(42)
    r_i = rng.uniform(0.5, 1.5, 20)
    r_j = rng.uniform(0.5, 1.5, 20)
    result = GloScope_py.kl_divergence(r_i=r_i, r_j=r_j, m_i=20, m_j=25, d=10)
    assert np.isscalar(result) or result.ndim == 0


# ---------------------------------------------------------------------------
# _get_data branch coverage
# ---------------------------------------------------------------------------


def test_get_data_from_obsm(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.obsm["X_pca"] = np.random.default_rng(0).normal(size=(adata.n_obs, 10))
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X_pca")
    method.prepare_anndata(adata)
    with pytest.warns(UserWarning, match="adata.obsm"):
        data = method._get_data()
    assert data.shape == (adata.n_obs, 10)


def test_get_data_from_layers(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.layers["counts"] = adata.X.copy()
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="counts")
    method.prepare_anndata(adata)
    with pytest.warns(UserWarning, match="adata.layers"):
        data = method._get_data()
    assert data.shape == adata.shape


def test_get_data_invalid_layer_raises(synthetic_adata):
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="nonexistent")
    method.prepare_anndata(synthetic_adata.copy())
    with pytest.raises(ValueError, match="Cannot find layer"):
        method._get_data()


# ---------------------------------------------------------------------------
# Pseudobulk with non-default aggregate / distance parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aggregate,dist", [("median", "euclidean"), ("sum", "cosine")])
def test_pseudobulk_non_default_params(aggregate, dist, synthetic_adata):
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True, aggregate=aggregate, dist=dist)
    assert distances.shape == (n_samples, n_samples)
    assert np.allclose(distances, distances.T)


# ---------------------------------------------------------------------------
# SampleRepresentationMethod.embed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embed_method", ["MDS", "TSNE", "UMAP"])
def test_embed_produces_2d_coordinates(embed_method, synthetic_adata):
    if embed_method == "UMAP":
        pytest.importorskip("umap")
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(adata)
    method.calculate_distance_matrix(force=True)

    coords = method.embed(method=embed_method)

    assert coords.shape == (n_samples, 2)
    assert embed_method in method.embeddings


def test_embed_unsupported_method_raises(synthetic_adata):
    adata = synthetic_adata.copy()
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(adata)
    method.calculate_distance_matrix(force=True)

    with pytest.raises(ValueError, match="not supported"):
        method.embed(method="PCA")


# ---------------------------------------------------------------------------
# SampleRepresentationMethod.to_adata
# ---------------------------------------------------------------------------


def test_to_adata_returns_anndata_with_correct_shape(synthetic_adata):
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(adata)
    method.calculate_distance_matrix(force=True)

    samples_adata = method.to_adata()

    assert isinstance(samples_adata, AnnData)
    assert samples_adata.n_obs == n_samples


# ---------------------------------------------------------------------------
# correlate_composition
# ---------------------------------------------------------------------------


def _make_meta_adata(adata, target_key="target", seed=0):
    """Create a minimal sample-level AnnData with a numeric target column."""
    sample_ids = adata.obs[SAMPLE_KEY].unique()
    rng = np.random.default_rng(seed)
    return AnnData(obs=pd.DataFrame({target_key: rng.normal(size=len(sample_ids))}, index=sample_ids))


def test_correlate_composition_returns_expected_columns(synthetic_adata):
    adata = synthetic_adata.copy()
    meta = _make_meta_adata(adata)

    result = correlate_composition(meta, adata, SAMPLE_KEY, CELL_KEY, target="target")

    assert isinstance(result, pd.DataFrame)
    assert {"correlation", "p_value", "p_value_adj", "-log_p_value_adj"}.issubset(result.columns)
    assert len(result) == adata.obs[CELL_KEY].nunique()


def test_correlate_composition_pearson(synthetic_adata):
    adata = synthetic_adata.copy()
    meta = _make_meta_adata(adata)

    result = correlate_composition(meta, adata, SAMPLE_KEY, CELL_KEY, target="target", method="pearson")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == adata.obs[CELL_KEY].nunique()


def test_correlate_composition_raises_for_invalid_method(synthetic_adata):
    adata = synthetic_adata.copy()
    meta = _make_meta_adata(adata)

    with pytest.raises(ValueError, match="spearman"):
        correlate_composition(meta, adata, SAMPLE_KEY, CELL_KEY, target="target", method="kendall")


# ---------------------------------------------------------------------------
# correlate_cell_type_expression
# ---------------------------------------------------------------------------


def test_correlate_cell_type_expression_returns_expected_columns(synthetic_adata):
    adata = synthetic_adata.copy()
    meta = _make_meta_adata(adata)

    result = correlate_cell_type_expression(
        meta, adata, SAMPLE_KEY, CELL_KEY, target="target", layer="X", min_sample_size=0
    )

    assert isinstance(result, pd.DataFrame)
    assert {"cell_type", "gene_name", "correlation", "p_value", "n_observations", "-log_p_value_adj"}.issubset(
        result.columns
    )


def test_correlate_cell_type_expression_raises_for_invalid_method(synthetic_adata):
    adata = synthetic_adata.copy()
    meta = _make_meta_adata(adata)

    with pytest.raises(ValueError, match="spearman"):
        correlate_cell_type_expression(
            meta, adata, SAMPLE_KEY, CELL_KEY, target="target", layer="X", min_sample_size=0, method="kendall"
        )


# ---------------------------------------------------------------------------
# create_colormap
# ---------------------------------------------------------------------------


def test_create_colormap_returns_series_with_unique_color_per_category(synthetic_adata):
    result = create_colormap(synthetic_adata.obs, "cell_type")
    assert len(result) == len(synthetic_adata.obs)
    assert result.nunique() == synthetic_adata.obs["cell_type"].nunique()


# ---------------------------------------------------------------------------
# GroupedPseudobulk with non-default aggregate / distance parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aggregate,dist", [("median", "euclidean"), ("sum", "cosine")])
def test_grouped_pseudobulk_non_default_params(aggregate, dist, synthetic_adata):
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()
    method = GroupedPseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True, aggregate=aggregate, dist=dist)
    assert distances.shape == (n_samples, n_samples)
    assert np.allclose(distances, distances.T)


# ---------------------------------------------------------------------------
# SampleRepresentationMethod._move_layer_to_X
# ---------------------------------------------------------------------------


def test_move_layer_to_x_returns_same_adata_when_layer_is_x(synthetic_adata):
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(synthetic_adata.copy())
    result = method._move_layer_to_X()
    assert result is method.adata


def test_move_layer_to_x_moves_layer_data_to_x(synthetic_adata):
    adata = synthetic_adata.copy()
    layer_data = adata.X.copy() * 2
    adata.layers["scaled"] = layer_data
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="scaled")
    method.prepare_anndata(adata)
    with pytest.warns(UserWarning, match="adata.layers"):
        result = method._move_layer_to_X()
    assert result is not method.adata
    assert np.array_equal(result.X, layer_data)
    assert "X_old" in result.obsm


# ---------------------------------------------------------------------------
# SampleRepresentationMethod.predict_metadata
# ---------------------------------------------------------------------------


def _prepare_method_with_distances(adata):
    """Return a Pseudobulk method with distances already computed."""
    method = Pseudobulk(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, layer="X")
    method.prepare_anndata(adata)
    method.calculate_distance_matrix(force=True)
    return method


def test_predict_metadata_classification(synthetic_adata):
    adata = synthetic_adata.copy()
    method = _prepare_method_with_distances(adata)
    n_samples = len(method.samples)
    metadata = pd.DataFrame({"condition": ["A", "B"] * (n_samples // 2)}, index=method.samples)

    y_true, y_pred = method.predict_metadata("condition", metadata=metadata)

    assert len(y_true) == n_samples
    assert len(y_pred) == n_samples
    assert set(y_pred).issubset({"A", "B"})


def test_predict_metadata_regression(synthetic_adata):
    adata = synthetic_adata.copy()
    method = _prepare_method_with_distances(adata)
    n_samples = len(method.samples)
    metadata = pd.DataFrame({"score": np.arange(n_samples, dtype=float)}, index=method.samples)

    y_true, y_pred = method.predict_metadata("score", metadata=metadata, task="regression")

    assert len(y_true) == n_samples
    assert len(y_pred) == n_samples


def test_predict_metadata_invalid_task_raises(synthetic_adata):
    adata = synthetic_adata.copy()
    method = _prepare_method_with_distances(adata)
    n_samples = len(method.samples)
    metadata = pd.DataFrame({"target": [0] * n_samples}, index=method.samples)

    with pytest.raises(ValueError, match="not supported"):
        method.predict_metadata("target", metadata=metadata, task="ranking")


# ---------------------------------------------------------------------------
# GloScope_py with n_components
# ---------------------------------------------------------------------------


def test_gloscope_py_with_n_components(pbmc3k_adata):
    pytest.importorskip("pynndescent")
    adata = pbmc3k_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = GloScope_py(sample_key=SAMPLE_KEY, cell_group_key=PBMC_CELL_KEY, layer="X_pca", n_components=5)
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    assert distances.shape == (n_samples, n_samples)
    assert np.allclose(distances, distances.T, atol=1e-5)


# ---------------------------------------------------------------------------
# WassersteinTSNE recomputation with different covariance_weight
# ---------------------------------------------------------------------------


def test_wasserstein_tsne_warns_when_recomputing_with_different_weight(pbmc3k_adata):
    pytest.importorskip("WassersteinTSNE")
    adata = pbmc3k_adata.copy()

    method = WassersteinTSNE(
        sample_key=SAMPLE_KEY,
        cell_group_key=PBMC_CELL_KEY,
        replicate_key=PBMC_CELL_KEY,
        layer="X_pca",
    )
    method.prepare_anndata(adata)
    method.calculate_distance_matrix(covariance_weight=0.5)

    with pytest.warns(UserWarning, match="Rewriting"):
        new_distances = method.calculate_distance_matrix(covariance_weight=0.3)

    assert new_distances.shape == (adata.obs[SAMPLE_KEY].nunique(), adata.obs[SAMPLE_KEY].nunique())
    assert adata.uns["wasserstein_covariance_weight"] == 0.3
