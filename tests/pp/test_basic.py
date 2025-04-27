import numpy as np
import pandas as pd

from patient_representation.pp.basic import (
    calculate_cell_qc_metrics,
    calculate_compositional_metrics,
    calculate_n_cells_per_sample,
    extract_metadata,
    fill_nan_distances,
    filter_small_cell_groups,
    filter_small_samples,
    is_count_data,
    subsample,
)


def test_calculate_compositional_metrics(toy_adata):
    result = calculate_compositional_metrics(
        toy_adata, sample_key="sample", composition_keys=["cell_type"], normalize_to=100
    )

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."
    assert not result.empty, "The result should not be empty."
    assert set(result.index) == set(
        toy_adata.obs["sample"].unique()
    ), "The result index should match the unique sample values in toy_adata."
    assert all(
        col.startswith("cell_type_") for col in result.columns
    ), "The result should have column names indicating cell type composition."
    assert np.isclose(
        result.loc["sample1"].sum(), 100
    ), "The sum of compositional metrics for sample1 should be equal to 100."
    assert np.isclose(
        result.loc["sample2"].sum(), 100
    ), "The sum of compositional metrics for sample2 should be equal to 100."


def test_extract_metadata(toy_adata):
    result = extract_metadata(toy_adata, sample_key="sample", columns=["cell_type"], samples=["sample1", "sample2"])

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."
    assert result.shape[0] == 2, "The result should contain metadata for 2 samples."
    assert "cell_type" in result.columns, "The result should contain the cell_type column."
    assert set(result.index) == {"sample1", "sample2"}, "The index should match the provided sample list."
    assert result.loc["sample1", "cell_type"] == "typeA", "The cell type for sample1 should be 'typeA'."


def test_calculate_cell_qc_metrics(toy_adata):
    toy_adata.obs["QC_ngenes"] = [10, 20, 30]
    toy_adata.obs["QC_total_UMI"] = [100, 200, 300]
    result = calculate_cell_qc_metrics(
        toy_adata, sample_key="sample", cell_qc_vars=["QC_ngenes", "QC_total_UMI"], agg_function=np.median
    )

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."
    assert not result.empty, "The result should not be empty."
    assert "median_QC_ngenes" in result.columns, "The result should contain the median_QC_ngenes column."
    assert "median_QC_total_UMI" in result.columns, "The result should contain the median_QC_total_UMI column."
    assert result.loc["sample1", "median_QC_ngenes"] == 20, "The median QC_ngenes for sample1 should be 20."
    assert result.loc["sample2", "median_QC_total_UMI"] == 200, "The median QC_total_UMI for sample2 should be 200."


def test_calculate_n_cells_per_sample(toy_adata):
    result = calculate_n_cells_per_sample(toy_adata, sample_key="sample")

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."
    assert set(result.index) == set(
        toy_adata.obs["sample"].unique()
    ), "The result should contain the sample as the index."
    assert "n_cells" in result.columns, "The result should contain the n_cells column."
    assert result.loc["sample1", "n_cells"] == 2, "Sample 'sample1' should have 2 cells."
    assert result.loc["sample2", "n_cells"] == 1, "Sample 'sample2' should have 1 cell."


def test_filter_small_samples(toy_adata):
    filtered_adata = filter_small_samples(toy_adata, sample_key="sample", sample_size_threshold=2)

    assert (
        filtered_adata.shape[0] < toy_adata.shape[0]
    ), "The filtered AnnData object should have fewer cells if samples do not meet the threshold."
    assert (
        "sample2" not in filtered_adata.obs["sample"].values
    ), "Sample 'sample2' should be removed due to the size threshold."
    assert "sample1" in filtered_adata.obs["sample"].values, "Sample 'sample1' should still be present."


def test_filter_small_cell_groups(toy_adata):
    filtered_adata = filter_small_cell_groups(
        toy_adata, sample_key="sample", cell_group_key="cell_type", cluster_size_threshold=2
    )

    assert (
        filtered_adata.shape[0] < toy_adata.shape[0]
    ), "The filtered AnnData object should have fewer cells if cell groups do not meet the threshold."
    assert (
        "typeB" not in filtered_adata.obs["cell_type"].values
    ), "Cell type 'typeB' should be removed due to the cluster size threshold."
    assert (
        "typeA" not in filtered_adata.obs["cell_type"].values
    ), "Cell type 'typeA' should be removed because Sample 'sample2' does not have any typeA group ."


def test_subsample(toy_adata):
    subsampled_adata = subsample(toy_adata, obs_category_col="cell_type", min_samples_per_category=1, fraction=0.5)

    assert (
        subsampled_adata.shape[0] <= toy_adata.shape[0]
    ), "The subsampled AnnData object should have fewer or equal number of cells."
    assert (
        "typeA" in subsampled_adata.obs["cell_type"].values
    ), "The subsampled data should still contain cells from 'typeA'."
    assert (
        subsampled_adata.shape[0] >= 1
    ), "Since fraction is 0.5, subsampled AnnData should have roughly half the cells from typeA."


def test_is_count_data():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = is_count_data(matrix)

    assert result, "The matrix should be identified as count data."

    non_count_matrix = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    result = is_count_data(non_count_matrix)
    assert not result, "The matrix should not be identified as count data."


def test_fill_nan_distances():
    distances = np.array([[0, np.nan, 3], [np.nan, 0, 2], [3, 2, 0]])
    filled_distances = fill_nan_distances(distances, n_max_distances=1)

    assert not np.isnan(filled_distances).any(), "There should be no NaN values in the filled distances."
    assert (
        filled_distances.shape == distances.shape
    ), "The filled distances should have the same shape as the original distances."
    assert filled_distances[0, 1] == filled_distances[1, 0], "The filled NaN values should be symmetric."
