from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def prepare_data_for_phemd(adata, sample_col, n_top_var_genes: int = 100):
    """Convert the expression data to the input format of PhEMD (R implementation)

    Returns
    -------
    all_expression_data : list
        Expression data of G genes for the cells of each of S samples
    all_genes : list[str]
        Names of the genes for the expression data
    selected_genes : list[str]
        Subset of genes to use
    samples_names : list[str]
        List of S names for the samples
    """
    top_variance = adata.var["variances"].sort_values(ascending=False)[:n_top_var_genes]
    selected_genes = top_variance.index

    import scipy.sparse as sp
    if sp.issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    samples_names = adata.obs[sample_col]

    return expression_data, adata.var_names, selected_genes, samples_names


def convert_cell_types_to_phemd_format(
    adata, cell_type_col, sample_col, output_dir="./", cell_types=None, n_top_var_genes=100
):
    """Converts `adata` to the tables required by PhEMD (R implementation) and saves them to `output_dir`"""
    if cell_types is None:
        cell_types = adata.obs[cell_type_col].unique()

    for cell_type in cell_types:
        cell_type_adata = adata[adata.obs[cell_type_col] == cell_type]
        all_expression_data, all_genes, selected_genes, samples_names = prepare_data_for_phemd(
            cell_type_adata, sample_col, n_top_var_genes
        )

        cell_type_dir = Path(output_dir) / cell_type
        cell_type_dir.mkdir(exist_ok=True)

        pd.DataFrame(all_expression_data).to_csv(cell_type_dir / "expression.csv", index=False, header=False)
        pd.DataFrame(all_genes).to_csv(cell_type_dir / "all_genes.csv", index=False, header=False)
        pd.DataFrame(selected_genes).to_csv(cell_type_dir / "selected_genes.csv", index=False, header=False)
        pd.DataFrame(samples_names).to_csv(cell_type_dir / "samples.csv", index=False, header=False)


def calculate_compositional_metrics(adata, sample_key, composition_keys, normalize_to: int = 100) -> pd.DataFrame:
    """
    Calculate compositional metrics for the given AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    sample_key : str
        Key for the sample information in `adata.obs`
    composition_keys : list[str]
        List of columns from `adata.obs` representing the composition categories (e.g. cell type)
    normalize_to : int = 100
        Value to which the compositional metrics will be normalized. Default is 100

    Returns
    -------
    compositional_metrics : pandas.DataFrame
        DataFrame containing compositional metrics. Rows are samples, and columns
        are categories from each of `composition_keys`. Values are fractions of
        categories in samples

    Examples
    --------
    >>> example = sc.AnnData(
            X=np.random.normal(size=(4, 2)),
            obs=pd.DataFrame(
                {"sample": ["a", "a", "b", "b"],
                 "cell_type": ["A", "B", "A", "A"]})
            )
    >>> calculate_compositional_metrics(example, sample_key="sample", composition_keys=["cell_type"])
    cell_type  cell_type_A  cell_type_B
    sample
    a                 50.0         50.0
    b                100.0          0.0
    """
    compositional_metrics = []

    for col in composition_keys:
        # Create table of counts of cells in each sample per category
        col_proportions = pd.crosstab(
            index=adata.obs[sample_key],
            columns=adata.obs[col],
            normalize="index",  # Sum by sample equals to 1
        )

        # Add name of the original column to new columns
        # E.g. if there were a column "cell_type" with categories "B" and "T"
        # In the resulting data frame there will be columns "cell_type_B" and "cell_type_T"
        new_col_names = {category: f"{col}_{category}" for category in col_proportions.columns}
        col_proportions = col_proportions.rename(columns=new_col_names)
        col_proportions *= normalize_to

        compositional_metrics.append(col_proportions)

    compositional_metrics = pd.concat(compositional_metrics, axis=1)

    return compositional_metrics


def extract_metadata(adata: sc.AnnData, sample_key: str, columns: list, samples: list = None) -> pd.DataFrame:
    """
    Return a dataframe with requested `columns` in the correct rows order.

    Parameters
    ----------
    - adata (sc.AnnData): The AnnData object containing the data.
    - sample_key (str): The key identifying the sample in the observation metadata.
    - columns (list): A list of column names to extract from the observation metadata.

    Returns
    -------
    - pd.DataFrame: A dataframe with the requested columns, indexed by the sample key.
    """
    if samples is None:
        samples = adata.obs[sample_key].unique()

    metadata = adata.obs[[sample_key, *columns]].drop_duplicates()

    # Check if the sample key is also in columns to avoid reindexing errors
    need_to_rename_sample_key = sample_key in columns

    # To avoid error, we rename column with sample key, reindex dataframe, and then rename sample column back
    if need_to_rename_sample_key:
        # Rename the first column with sample key to sample_key_dupl
        metadata.columns = [sample_key + "_dupl"] + list(metadata.columns[1:])

    metadata = metadata.set_index(sample_key)

    if need_to_rename_sample_key:
        metadata.rename(columns={sample_key + "_dupl": sample_key}, inplace=True)

    return metadata.loc[samples]


def calculate_cell_qc_metrics(adata, sample_key, cell_qc_vars, agg_function=np.median) -> pd.DataFrame:
    """
    Calculate agregated cell quality control metrics for the given AnnData object

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    sample_key : str
        Key for the sample information in `adata.obs`
    cell_qc_vars: list[str]
        List of column keys representing the cell QC variables. For example, number of genes per cell
    agg_function: Callable = numpy.median
        Aggregation function to use for aggregating cell QC metrics. Default is numpy.median

    Returns
    -------
    cells_qc_aggregated : pandas.DataFrame
        DataFrame with samples in rows and aggregated QC metrics in columns

    Examples
    --------
    >>> calculate_cell_qc_metrics(adata, sample_key="scRNASeq_sample_ID", cell_qc_vars=["QC_ngenes", "QC_total_UMI"])
                        median_QC_ngenes  median_QC_total_UMI
    scRNASeq_sample_ID
    G05061-Ja005E-PBCa            1112.0               3150.0
    G05064-Ja005E-PBCa             982.5               2955.0
    """
    new_col_names = {col_name: agg_function.__name__ + "_" + col_name for col_name in cell_qc_vars}

    metadata = adata.obs[[sample_key, *cell_qc_vars]].groupby(by=sample_key)

    cells_qc_aggregated = metadata.aggregate(agg_function)
    cells_qc_aggregated = cells_qc_aggregated.rename(columns=new_col_names)

    return cells_qc_aggregated


def calculate_n_cells_per_sample(adata, sample_key) -> pd.DataFrame:
    """
    Calculate the number of cells per sample in the given AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    sample_key : str
        Key for the sample information in `adata.obs`

    Returns
    -------
    cell_counts : pandas.DataFrame
        DataFrame containing the number of cells per sample in the columns "n_cells"
    """
    cell_counts = pd.DataFrame(adata.obs[sample_key].value_counts())
    cell_counts.columns = ["n_cells"]
    return cell_counts


def filter_small_samples(adata, sample_key, sample_size_threshold: int = 300):
    """Leave only samples with not less than `sample_size_threshold` cells"""
    sample_size_counts = adata.obs[sample_key].value_counts()
    small_samples = sample_size_counts[sample_size_counts < sample_size_threshold].index
    filtered_samples = set(adata.obs[sample_key]) - set(small_samples)
    print(len(small_samples), "samples removed:", ", ".join(small_samples))

    adata = adata[adata.obs[sample_key].isin(filtered_samples)].copy()

    return adata


def filter_small_cell_groups(adata, sample_key, cell_group_key, cluster_size_threshold: int = 5):
    """Leave only cell groups with not less than `cluster_size_threshold` cells"""
    cells_counts = adata.obs[[sample_key, cell_group_key]].value_counts().reset_index(name="count")

    # This step does not filter cell types with 0 counts
    small_cell_types = cells_counts.loc[cells_counts["count"] < cluster_size_threshold, cell_group_key].unique()
    small_cell_types = set(small_cell_types)

    if cluster_size_threshold > 0:
        # Add cell types with 0 counts in some samples
        for sample in adata.obs[sample_key].unique():
            for cell_type in adata.obs[cell_group_key].unique():
                sample_cells = adata[(adata.obs[sample_key] == sample) & (adata.obs[cell_group_key] == cell_type)]
                if not sample_cells:
                    small_cell_types.add(cell_type)

    filtered_cell_types = set(adata.obs[cell_group_key]) - set(small_cell_types)
    print(len(small_cell_types), "cell types removed:", ", ".join(small_cell_types))

    adata = adata[adata.obs[cell_group_key].isin(filtered_cell_types)].copy()

    return adata


def subsample(adata, obs_category_col: str, min_samples_per_category: int, fraction=None, n_obs=None):
    """Subsample cells from each category in `obs_category_col` to have at least `min_samples_per_category` cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing cells.
    obs_category_col : str
        Name of the column in `adata.obs` containing categories to subsample.
    min_samples_per_category : int
        Minimum number of cells per category.
    fraction : float or None
        Fraction of cells to take from each category. If `None`, `n_obs` must be set.
    n_obs : int or None
        Number of cells to take from each category. If `None`, `fraction` must be set.

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    subsample_idxs = []

    assert fraction is None or 0 < fraction <= 1, "`fraction` must be a number between 0 and 1"
    if fraction is None:
        assert n_obs is not None and int(n_obs), "`n_obs` must be an integer number or `fraction` must be set"

    for level in adata.obs[obs_category_col].unique():
        level_cells = adata.obs[obs_category_col] == level
        obs_per_level = sum(level_cells)
        level_idxs = np.where(level_cells)[0]

        if obs_per_level <= min_samples_per_category:
            # Take all cells from this level
            subsample_idxs.extend(level_idxs)
        else:
            if fraction is not None:
                n_cells = int(fraction * obs_per_level)
            else:
                n_cells = int(n_obs)

            selected_cells_idxs = np.random.choice(level_idxs, size=n_cells, replace=False)
            subsample_idxs.extend(selected_cells_idxs)

    return adata[subsample_idxs]


def is_count_data(matrix, window_size=10000) -> bool:
    """Ensure that `matrix` only contains integers"""
    from scipy.sparse import issparse

    if issparse(matrix):
        return np.all(matrix[:window_size, :window_size].data % 1 == 0)

    return np.all(matrix[:window_size, :window_size] % 1 == 0)


def fill_nan_distances(distances, n_max_distances=5):
    """Fill NaN values in `distances` with maximum distance multiplied by `n_max_distances`"""
    distances = distances.copy()
    nans = np.isnan(distances)
    max_distance = distances[~nans].max()

    distances[nans] = n_max_distances * max_distance

    return distances



###
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