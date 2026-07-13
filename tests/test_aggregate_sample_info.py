import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from patpy.pp import aggregate_sample_info

SAMPLE_KEY = "sample_id"
CELL_TYPE_KEY = "cell_type"


@pytest.fixture
def adata_with_metadata(synthetic_adata):
    """Synthetic AnnData with sample-level metadata, an embedding and a layer."""
    adata = synthetic_adata.copy()
    disease = {"sample_0": "healthy", "sample_1": "healthy", "sample_2": "healthy"}
    adata.obs["disease"] = [disease.get(sample, "sick") for sample in adata.obs[SAMPLE_KEY]]
    adata.obs["age"] = adata.obs[SAMPLE_KEY].str.removeprefix("sample_").astype(int) * 10
    adata.obsm["X_pca"] = np.arange(adata.n_obs * 5, dtype=float).reshape(adata.n_obs, 5)
    adata.layers["counts"] = adata.X.copy()
    return adata


def _n_samples(adata):
    return adata.obs[SAMPLE_KEY].nunique()


# Without optional keys the result is an empty object with one row per sample.
def test_only_sample_key(adata_with_metadata):
    meta_adata = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY)

    assert meta_adata.n_obs == _n_samples(adata_with_metadata)
    assert meta_adata.n_vars == 0
    assert list(meta_adata.obs.columns) == []
    assert list(meta_adata.obs_names) == [str(s) for s in adata_with_metadata.obs[SAMPLE_KEY].unique()]
    assert meta_adata.obsm == {}
    assert dict(meta_adata.layers) == {}


# .X must hold the mean pseudobulk of the requested layer, with gene names kept.
def test_pseudobulk_from_X(adata_with_metadata):
    meta_adata = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, layer="X")

    assert meta_adata.shape == (_n_samples(adata_with_metadata), adata_with_metadata.n_vars)
    assert list(meta_adata.var_names) == list(adata_with_metadata.var_names)

    for sample in meta_adata.obs_names:
        expected = adata_with_metadata[adata_with_metadata.obs[SAMPLE_KEY] == sample].X.mean(axis=0)
        np.testing.assert_allclose(meta_adata[sample].X.flatten(), expected)


# An .obsm embedding is a valid source and gets positional feature names.
def test_pseudobulk_from_obsm(adata_with_metadata):
    meta_adata = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, layer="X_pca")

    assert meta_adata.n_vars == 5
    assert list(meta_adata.var_names) == [f"X_pca_{i}" for i in range(5)]

    sample = meta_adata.obs_names[0]
    cells = adata_with_metadata.obs[SAMPLE_KEY] == sample
    np.testing.assert_allclose(meta_adata[sample].X.flatten(), adata_with_metadata.obsm["X_pca"][cells].mean(axis=0))


# A .layers entry is a valid source as well.
def test_pseudobulk_from_layer(adata_with_metadata):
    from_layer = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, layer="counts")
    from_x = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, layer="X")

    np.testing.assert_allclose(from_layer.X, from_x.X)


# Sparse .X is aggregated to the same values as its dense counterpart.
def test_pseudobulk_sparse_matches_dense(adata_with_metadata):
    dense = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, layer="X")

    sparse_adata = adata_with_metadata.copy()
    sparse_adata.X = sp.csr_matrix(sparse_adata.X)
    sparse = aggregate_sample_info(sparse_adata, sample_key=SAMPLE_KEY, layer="X")

    np.testing.assert_allclose(np.asarray(sparse.X), dense.X)


# Aggregation function is configurable.
def test_aggregate_sum(adata_with_metadata):
    meta_adata = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, layer="X", aggregate="sum")

    sample = meta_adata.obs_names[0]
    cells = adata_with_metadata.obs[SAMPLE_KEY] == sample
    np.testing.assert_allclose(meta_adata[sample].X.flatten(), adata_with_metadata[cells].X.sum(axis=0))


# Cell-type composition lands in .obsm, sums to one, and matches the counts.
def test_cell_type_composition(adata_with_metadata):
    meta_adata = aggregate_sample_info(
        adata_with_metadata, sample_key=SAMPLE_KEY, cell_type_key=CELL_TYPE_KEY, layer="X"
    )

    composition = meta_adata.obsm["cell_type_composition"]
    assert isinstance(composition, pd.DataFrame)
    assert list(composition.index) == list(meta_adata.obs_names)
    assert set(composition.columns) == set(adata_with_metadata.obs[CELL_TYPE_KEY].unique())
    np.testing.assert_allclose(composition.sum(axis=1), 1)

    expected = pd.crosstab(
        adata_with_metadata.obs[SAMPLE_KEY],
        adata_with_metadata.obs[CELL_TYPE_KEY],
        normalize="index",
    )
    for sample in meta_adata.obs_names:
        for cell_type in composition.columns:
            assert composition.loc[sample, cell_type] == pytest.approx(expected.loc[sample, cell_type])


# Composition can be requested without any expression aggregation.
def test_composition_without_layer(adata_with_metadata):
    meta_adata = aggregate_sample_info(adata_with_metadata, sample_key=SAMPLE_KEY, cell_type_key=CELL_TYPE_KEY)

    assert meta_adata.n_vars == 0
    assert meta_adata.obsm["cell_type_composition"].shape == (
        _n_samples(adata_with_metadata),
        adata_with_metadata.obs[CELL_TYPE_KEY].nunique(),
    )


# Per-cell-type pseudobulk layers are named after the cell types and match manual means.
def test_cell_type_pseudobulk_layers(adata_with_metadata):
    meta_adata = aggregate_sample_info(
        adata_with_metadata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        layer="X",
        cell_type_pseudobulk=True,
    )

    cell_types = adata_with_metadata.obs[CELL_TYPE_KEY].unique()
    assert set(meta_adata.layers) == {f"{cell_type}_pseudobulk" for cell_type in cell_types}

    for cell_type in cell_types:
        layer = meta_adata.layers[f"{cell_type}_pseudobulk"]
        assert layer.shape == meta_adata.shape

        for i, sample in enumerate(meta_adata.obs_names):
            cells = (adata_with_metadata.obs[SAMPLE_KEY] == sample) & (
                adata_with_metadata.obs[CELL_TYPE_KEY] == cell_type
            )
            np.testing.assert_allclose(layer[i], adata_with_metadata[cells].X.mean(axis=0))


# Samples missing a cell type get `fill_value` in that cell type's layer.
def test_cell_type_pseudobulk_missing_combination(adata_with_metadata):
    adata = adata_with_metadata.copy()
    # Remove every ct_c cell of sample_0 so that combination has no cells left
    keep = ~((adata.obs[SAMPLE_KEY] == "sample_0") & (adata.obs[CELL_TYPE_KEY] == "ct_c"))
    adata = adata[keep].copy()

    meta_adata = aggregate_sample_info(
        adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        layer="X",
        cell_type_pseudobulk=True,
        fill_value=np.nan,
    )

    missing_row = meta_adata["sample_0"].layers["ct_c_pseudobulk"]
    assert np.isnan(missing_row).all()
    assert meta_adata.obsm["cell_type_composition"].loc["sample_0", "ct_c"] == 0

    other_row = meta_adata["sample_1"].layers["ct_c_pseudobulk"]
    assert not np.isnan(other_row).any()


# Sample metadata ends up in .obs, aligned with the sample order.
def test_metadata_cols(adata_with_metadata):
    meta_adata = aggregate_sample_info(
        adata_with_metadata,
        sample_key=SAMPLE_KEY,
        layer="X",
        metadata_cols=["disease", "age"],
    )

    assert list(meta_adata.obs.columns) == ["disease", "age"]
    assert meta_adata.obs.loc["sample_0", "disease"] == "healthy"
    assert meta_adata.obs.loc["sample_5", "disease"] == "sick"
    assert meta_adata.obs.loc["sample_3", "age"] == 30


# Everything at once: pseudobulk, composition, per-cell-type layers and metadata.
def test_all_options_together(adata_with_metadata):
    meta_adata = aggregate_sample_info(
        adata_with_metadata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        layer="X_pca",
        metadata_cols=["disease"],
        cell_type_pseudobulk=True,
    )

    n_samples = _n_samples(adata_with_metadata)
    assert meta_adata.shape == (n_samples, 5)
    assert len(meta_adata.layers) == adata_with_metadata.obs[CELL_TYPE_KEY].nunique()
    assert meta_adata.obsm["cell_type_composition"].shape[0] == n_samples
    assert list(meta_adata.obs.columns) == ["disease"]
    assert meta_adata.uns["aggregate_sample_info"]["layer"] == "X_pca"


# Sample order follows order of appearance in .obs and is consistent across slots.
def test_sample_order_is_consistent():
    adata = AnnData(
        np.arange(24, dtype=float).reshape(6, 4),
        obs=pd.DataFrame(
            {
                SAMPLE_KEY: ["b", "b", "a", "a", "c", "c"],
                CELL_TYPE_KEY: ["ct_a", "ct_b", "ct_a", "ct_a", "ct_b", "ct_b"],
                "disease": ["sick", "sick", "healthy", "healthy", "sick", "sick"],
            }
        ),
    )

    meta_adata = aggregate_sample_info(
        adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        layer="X",
        metadata_cols=["disease"],
        cell_type_pseudobulk=True,
    )

    assert list(meta_adata.obs_names) == ["b", "a", "c"]
    assert list(meta_adata.obs["disease"]) == ["sick", "healthy", "sick"]
    assert list(meta_adata.obsm["cell_type_composition"].index) == ["b", "a", "c"]
    # Sample "a" only has ct_a cells: rows 2 and 3
    np.testing.assert_allclose(meta_adata["a"].layers["ct_a_pseudobulk"].flatten(), adata.X[2:4].mean(axis=0))
    np.testing.assert_allclose(meta_adata["a"].X.flatten(), adata.X[2:4].mean(axis=0))


# Invalid keys and impossible combinations raise informative errors.
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_key": "missing"}, "sample_key"),
        ({"sample_key": SAMPLE_KEY, "cell_type_key": "missing"}, "cell_type_key"),
        ({"sample_key": SAMPLE_KEY, "layer": "missing"}, "layer"),
        ({"sample_key": SAMPLE_KEY, "cell_type_pseudobulk": True, "layer": "X"}, "cell_type_pseudobulk"),
        (
            {"sample_key": SAMPLE_KEY, "cell_type_pseudobulk": True, "cell_type_key": CELL_TYPE_KEY},
            "cell_type_pseudobulk",
        ),
    ],
)
def test_invalid_arguments(adata_with_metadata, kwargs, match):
    with pytest.raises(ValueError, match=match):
        aggregate_sample_info(adata_with_metadata, **kwargs)
