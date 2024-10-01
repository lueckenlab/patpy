import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


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

    expression_data = adata.X.toarray()
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


def save_scitd_input_files(adata, output_dir, sample_col, cell_type_col, metadata_columns=None, raw_counts_layer=None):
    """Save the input files for ScITD (R implementation)

    It requires:
    1. Raw counts matrix: sparse matrix with cells in rows and genes in columns
    2. Cells metadata: sample and cell type annotations, optionally other covariates:
    dataframe with columns "donors", "ctypes", and optionally others
    3. Gene names mapping: dataframe without column names with two columns containing
    Ensembl IDs in the first column and gene names in the second. This is typically
    stored in various ways in `adata.var`, so this function does not creates such file.
    Please, do it manually.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    output_dir : Union[str, Path]
        Directory to save the input files
    sample_col : str
        Column in `adata.obs` containing sample annotations
    cell_type_col : str
        Column in `adata.obs` containing cell type annotations
    metadata_columns : list[str] = None
        List of other columns in `adata.obs` to include in the metadata file
    raw_counts_layer : str = "X"
        Layer in `adata` containing raw counts
    """
    if raw_counts_layer is None:
        expression_data = adata.X
    elif raw_counts_layer in adata.layers:
        expression_data = adata.layers[raw_counts_layer]
    elif raw_counts_layer in adata.obsm:
        expression_data = adata.obsm[raw_counts_layer]
    else:
        raise ValueError(f"Layer {raw_counts_layer} not found in adata")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if expression_data is sparse, if not, convert it to sparse
    if not scipy.sparse.issparse(expression_data):
        warnings.warn(
            "Expression data is not sparse. Converting to sparse matrix. Please make sure you provided the correct raw counts layer",
            stacklevel=1,
        )
        expression_data = scipy.sparse.csr_matrix(expression_data)

    scipy.io.mmwrite(output_dir / "expression_data.mtx", expression_data)

    metadata_cols = [sample_col, cell_type_col]

    if metadata_columns is not None:
        metadata_cols.extend(metadata_columns)

    metadata = adata.obs[metadata_cols].rename(columns={sample_col: "donors", cell_type_col: "ctypes"})
    metadata.to_csv(output_dir / "metadata.csv", index=True)


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
            index=adata.obs[sample_key], columns=adata.obs[col], normalize="index"  # Sum by sample equals to 1
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


def filter_small_cell_types(adata, sample_key, cells_type_key, cluster_size_threshold: int = 5):
    """Leave only cell types with not less than `cluster_size_threshold` cells"""
    cells_counts = adata.obs[[sample_key, cells_type_key]].value_counts().reset_index(name="count")

    # This step does not filter cell types with 0 counts
    small_cell_types = cells_counts.loc[cells_counts["count"] < cluster_size_threshold, cells_type_key].unique()
    small_cell_types = set(small_cell_types)

    if cluster_size_threshold > 0:
        # Add cell types with 0 counts in some samples
        for sample in adata.obs[sample_key].unique():
            for cell_type in adata.obs[cells_type_key].unique():
                sample_cells = adata[(adata.obs[sample_key] == sample) & (adata.obs[cells_type_key] == cell_type)]
                if not sample_cells:
                    small_cell_types.add(cell_type)

    filtered_cell_types = set(adata.obs[cells_type_key]) - set(small_cell_types)
    print(len(small_cell_types), "cell types removed:", ", ".join(small_cell_types))

    adata = adata[adata.obs[cells_type_key].isin(filtered_cell_types)].copy()

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
