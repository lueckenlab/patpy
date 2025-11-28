import numpy as np
import pandas as pd

from patpy.pp.basic import (
    calculate_cell_qc_metrics,
    calculate_compositional_metrics,
    calculate_n_cells_per_sample,
    convert_cell_types_to_phemd_format,
    extract_metadata,
    fill_nan_distances,
    filter_small_cell_groups,
    filter_small_samples,
    is_count_data,
    prepare_data_for_phemd,
    subsample,
)

SAMPLE_KEY = "sample_id"
CELL_KEY = "cell_type"


# Verify PhEMD preparation returns expected components and ordering.
def test_prepare_data_for_phemd_shapes(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.var["variances"] = np.linspace(0, 1, adata.n_vars)

    expression_data, all_genes, selected_genes, sample_names = prepare_data_for_phemd(
        adata, sample_col=SAMPLE_KEY, n_top_var_genes=5
    )

    assert expression_data.shape == adata.X.shape
    assert list(all_genes) == list(adata.var_names)
    assert len(selected_genes) == 5
    assert list(sample_names) == list(adata.obs[SAMPLE_KEY])


# Ensure PhEMD conversion writes all required tables per cell type.
def test_convert_cell_types_to_phemd_format(tmp_path, synthetic_adata):
    adata = synthetic_adata.copy()
    adata.var["variances"] = np.linspace(0, 1, adata.n_vars)

    convert_cell_types_to_phemd_format(
        adata, cell_type_col=CELL_KEY, sample_col=SAMPLE_KEY, output_dir=tmp_path, n_top_var_genes=5
    )

    for cell_type in adata.obs[CELL_KEY].unique():
        cell_dir = tmp_path / cell_type
        assert cell_dir.exists()
        expression = pd.read_csv(cell_dir / "expression.csv", header=None)
        all_genes = pd.read_csv(cell_dir / "all_genes.csv", header=None)
        selected = pd.read_csv(cell_dir / "selected_genes.csv", header=None)
        samples = pd.read_csv(cell_dir / "samples.csv", header=None)

        assert not expression.empty
        assert len(all_genes) == adata.n_vars
        assert len(selected) == 5
        assert len(samples) == len(expression)


# Check compositional metrics aggregation per sample and category.
def test_calculate_compositional_metrics(synthetic_adata):
    result = calculate_compositional_metrics(
        synthetic_adata, sample_key=SAMPLE_KEY, composition_keys=[CELL_KEY], normalize_to=100
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == set(synthetic_adata.obs[SAMPLE_KEY].unique())
    assert all(col.startswith("cell_type_") for col in result.columns)
    for sample in synthetic_adata.obs[SAMPLE_KEY].unique():
        assert np.isclose(result.loc[sample].sum(), 100)


# Validate QC metric aggregation and column naming.
def test_calculate_cell_qc_metrics(synthetic_adata):
    synthetic_adata.obs["QC_ngenes"] = np.linspace(10, 70, synthetic_adata.n_obs, dtype=int)
    synthetic_adata.obs["QC_total_UMI"] = np.linspace(100, 400, synthetic_adata.n_obs, dtype=int)

    result = calculate_cell_qc_metrics(
        synthetic_adata,
        sample_key=SAMPLE_KEY,
        cell_qc_vars=["QC_ngenes", "QC_total_UMI"],
        agg_function=np.median,
    )

    assert "median_QC_ngenes" in result.columns
    assert "median_QC_total_UMI" in result.columns
    assert not result.isna().any().any()


# Confirm cell counts per sample are tallied correctly.
def test_calculate_n_cells_per_sample(synthetic_adata):
    result = calculate_n_cells_per_sample(synthetic_adata, sample_key=SAMPLE_KEY)

    assert set(result.index) == set(synthetic_adata.obs[SAMPLE_KEY])
    assert result["n_cells"].sum() == synthetic_adata.n_obs


# Ensure samples below size threshold are removed.
def test_filter_small_samples(synthetic_adata):
    adata = synthetic_adata.copy()
    small_sample = adata.obs[SAMPLE_KEY].unique()[0]
    small_mask = adata.obs[SAMPLE_KEY] == small_sample
    indices = np.flatnonzero(small_mask.to_numpy())
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    keep_mask[small_mask.to_numpy()] = False
    keep_mask[indices[:2]] = True
    adata = adata[keep_mask].copy()

    filtered = filter_small_samples(adata, sample_key=SAMPLE_KEY, sample_size_threshold=3)

    assert small_sample not in filtered.obs[SAMPLE_KEY].values
    assert filtered.obs[SAMPLE_KEY].nunique() == adata.obs[SAMPLE_KEY].nunique() - 1


# Ensure undersized or absent cell groups are removed.
def test_filter_small_cell_groups(synthetic_adata):
    filtered = filter_small_cell_groups(
        synthetic_adata, sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, cluster_size_threshold=2
    )

    assert set(filtered.obs[CELL_KEY].unique()) == set(synthetic_adata.obs[CELL_KEY].unique())


# Check subsampling respects per-category minimums and fraction sizing.
def test_subsample(synthetic_adata):
    subsampled = subsample(
        synthetic_adata, obs_category_col=CELL_KEY, min_samples_per_category=1, fraction=0.5
    )

    assert subsampled.shape[0] <= synthetic_adata.shape[0]
    assert all(ct in subsampled.obs[CELL_KEY].values for ct in synthetic_adata.obs[CELL_KEY].unique())


# Verify metadata extraction preserves order and handles duplicate sample key column.
def test_extract_metadata_with_sample_column(synthetic_adata):
    donor_condition = {sample: f"group_{i}" for i, sample in enumerate(synthetic_adata.obs[SAMPLE_KEY].unique())}
    synthetic_adata.obs["donor_condition"] = synthetic_adata.obs[SAMPLE_KEY].map(donor_condition)

    metadata = extract_metadata(synthetic_adata, sample_key=SAMPLE_KEY, columns=[SAMPLE_KEY, "donor_condition"])

    assert list(metadata.index) == list(synthetic_adata.obs[SAMPLE_KEY].unique())
    assert SAMPLE_KEY in metadata.columns
    assert "donor_condition" in metadata.columns


# Validate integer-only detection for count matrices.
def test_is_count_data(integer_matrix):
    assert is_count_data(integer_matrix)

    non_count_matrix = np.array([[1.1, 2.2], [3.3, 4.4]])
    assert not is_count_data(non_count_matrix)


# Confirm NaN distances are filled symmetrically with max-distance scaling.
def test_fill_nan_distances():
    distances = np.array([[0, np.nan, 3], [np.nan, 0, 2], [3, 2, 0]], dtype=float)
    filled = fill_nan_distances(distances, n_max_distances=1)

    assert not np.isnan(filled).any()
    assert filled.shape == distances.shape
    assert filled[0, 1] == filled[1, 0]
